# -*- coding: utf-8 -*-
"""
RetNet with TaylorExp (Based) feature map on Q/K.

Mirrors retnet_rebased.py but uses fla's TaylorFeatureMap instead of RebasedFeatureMap.
Replaces XPOS rotary embedding with TaylorFeatureMap.

Expanded dim: 1 + 2*feature_dim + feature_dim*(feature_dim-1)//2
  feature_dim=10 → 1 + 20 + 45 = 66 ≈ head_k_dim=64  (d_model=128, num_heads=2)

Retention kernel becomes:
  k(q_i, k_j) = φ_taylor(q_i) · φ_taylor(k_j) * γ^(i-j)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.modules.feature_map import TaylorFeatureMap

from zoology.mixers.retnet import _get_decay_mask
from zoology.mixers.gla_based import taylor_expanded_size


class RetNetBased(nn.Module):
    """
    RetNet with TaylorExp (Based) feature map on Q/K (no XPOS).

    Args:
        d_model:      hidden dimension
        num_heads:    number of retention heads
        feature_dim:  per-head input dim to TaylorFeatureMap;
                      expanded key dim = 1 + 2*d + d*(d-1)//2
        double_v_dim: double V dimension
        recurrent_threshold: use recurrent mode when seq_len <= this
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 2,
        feature_dim: int = 10,
        double_v_dim: bool = False,
        recurrent_threshold: int = 64,
        **kwargs,
    ) -> None:
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.expanded_dim = taylor_expanded_size(feature_dim)
        self.v_dim = d_model * 2 if double_v_dim else d_model
        self.head_v_dim = self.v_dim // num_heads
        self.recurrent_threshold = recurrent_threshold

        # γ: multi-scale decay, one per head
        gammas = 1 - torch.exp(
            torch.linspace(math.log(1 / 32), math.log(1 / 512), num_heads)
        )
        self.register_buffer("gammas", gammas)  # (H,)

        qk_proj_dim = feature_dim * num_heads
        self.q_proj = nn.Linear(d_model, qk_proj_dim, bias=False)
        self.k_proj = nn.Linear(d_model, qk_proj_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.v_dim, bias=False)

        self.feature_map_q = TaylorFeatureMap(head_dim=feature_dim)
        self.feature_map_k = TaylorFeatureMap(head_dim=feature_dim)

        self.g_proj = nn.Linear(d_model, self.v_dim, bias=False)
        self.o_proj = nn.Linear(self.v_dim, d_model, bias=False)
        self.group_norm = nn.GroupNorm(num_heads, self.v_dim)

    def _parallel_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = rearrange(q, "b t (h d) -> b t h d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b t h d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b t h d", h=self.num_heads)

        # TaylorFeatureMap: (B, T, H, feature_dim) → (B, T, H, expanded_dim)
        q = self.feature_map_q(q)
        k = self.feature_map_k(k)

        q = q.permute(0, 2, 1, 3)  # (B, H, T, expanded_dim)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)  # (B, H, T, head_v_dim)

        D = torch.stack([
            _get_decay_mask(T, self.gammas[h].item(), hidden_states.device)
            for h in range(self.num_heads)
        ])  # (H, T, T)

        scale = self.expanded_dim ** -0.5
        attn = torch.einsum("bhtd,bhsd->bhts", q, k) * scale
        attn = attn * D.unsqueeze(0)
        Y = torch.einsum("bhts,bhsd->bhtd", attn, v)

        return rearrange(Y, "b h t d -> b t (h d)")

    def _recurrent_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        device = hidden_states.device

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = rearrange(q, "b t (h d) -> b t h d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b t h d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b t h d", h=self.num_heads)

        q = self.feature_map_q(q)
        k = self.feature_map_k(k)

        S = torch.zeros(B, self.num_heads, self.expanded_dim, self.head_v_dim, device=device)
        outputs = []
        scale = self.expanded_dim ** -0.5

        for t in range(T):
            q_t = q[:, t]   # (B, H, expanded_dim)
            k_t = k[:, t]
            v_t = v[:, t]   # (B, H, head_v_dim)

            kv = torch.einsum("bhd,bhe->bhde", k_t, v_t)
            S = self.gammas.view(1, -1, 1, 1) * S + kv
            y_t = torch.einsum("bhd,bhde->bhe", q_t, S) * scale
            outputs.append(y_t)

        Y = torch.stack(outputs, dim=1)   # (B, T, H, head_v_dim)
        return rearrange(Y, "b t h d -> b t (h d)")

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        B, T, _ = hidden_states.shape

        Y = self._recurrent_forward(hidden_states) if T <= self.recurrent_threshold \
            else self._parallel_forward(hidden_states)

        Y = self.group_norm(Y.transpose(1, 2)).transpose(1, 2)
        G = F.silu(self.g_proj(hidden_states))
        return self.o_proj(G * Y)

    def state_size(self, **kwargs) -> int:
        return self.num_heads * self.expanded_dim * self.head_v_dim
