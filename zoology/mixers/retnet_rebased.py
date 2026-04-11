# -*- coding: utf-8 -*-
"""
RetNet with ReBased feature map on Q/K.

Original RetNet uses XPOS rotary embedding on Q/K and dot-product kernel.
Here we replace XPOS with the ReBased quadratic feature map:
  φ(x) = LayerNorm(x) → upper-triangular outer product → dim: feature_dim*(feature_dim+1)//2

The retention kernel becomes:
  k(q_i, k_j) = φ(q_i) · φ(k_j) * γ^(i-j)   (causal exponential decay preserved)

This combines:
  - RetNet's temporal decay (multi-scale γ per head)
  - ReBased's richer quadratic kernel approximation (vs plain dot-product)

Q/K are projected to feature_dim first, then expanded to expanded_dim by the feature map.
V and the output gate are unchanged from RetNet.

State size: num_heads * expanded_dim * head_v_dim
  With feature_dim=11: expanded=66 ≈ 64 (same as standard RetNet with d_model=128, H=2)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.modules.feature_map import RebasedFeatureMap
from zoology.mixers.feature_maps.rebased import rebased_expanded_size
from zoology.mixers.retnet import _get_decay_mask


class RetNetReBased(nn.Module):
    """
    RetNet with ReBased quadratic feature map on Q/K (no XPOS).

    Args:
        d_model:      hidden dimension
        num_heads:    number of retention heads (each gets a different γ)
        feature_dim:  per-head input dim to RebasedFeatureMap;
                      expanded key dim = feature_dim*(feature_dim+1)//2
        double_v_dim: double V dimension
        recurrent_threshold: use recurrent mode when seq_len <= this
        use_gamma / use_beta_fm / normalize: RebasedFeatureMap hyperparams
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 2,
        feature_dim: int = 11,
        double_v_dim: bool = False,
        recurrent_threshold: int = 64,
        use_gamma: bool = True,
        use_beta_fm: bool = True,
        normalize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.expanded_dim = rebased_expanded_size(feature_dim)
        self.v_dim = d_model * 2 if double_v_dim else d_model
        self.head_v_dim = self.v_dim // num_heads
        self.recurrent_threshold = recurrent_threshold

        # γ: multi-scale decay, one per head
        gammas = 1 - torch.exp(
            torch.linspace(math.log(1 / 32), math.log(1 / 512), num_heads)
        )
        self.register_buffer("gammas", gammas)  # (H,)

        # Q/K projected to feature_dim per head (BEFORE expansion)
        qk_proj_dim = feature_dim * num_heads
        self.q_proj = nn.Linear(d_model, qk_proj_dim, bias=False)
        self.k_proj = nn.Linear(d_model, qk_proj_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.v_dim, bias=False)

        # Separate feature maps for Q and K (each has own learnable γ/β)
        self.feature_map_q = RebasedFeatureMap(
            head_dim=feature_dim,
            use_gamma=use_gamma,
            use_beta=use_beta_fm,
            normalize=normalize,
        )
        self.feature_map_k = RebasedFeatureMap(
            head_dim=feature_dim,
            use_gamma=use_gamma,
            use_beta=use_beta_fm,
            normalize=normalize,
        )

        # Output gate + projection
        self.g_proj = nn.Linear(d_model, self.v_dim, bias=False)
        self.o_proj = nn.Linear(self.v_dim, d_model, bias=False)

        # GroupNorm over v_dim channels
        self.group_norm = nn.GroupNorm(num_heads, self.v_dim)

    # ── parallel mode ──────────────────────────────────────────────────────────
    def _parallel_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape

        q = self.q_proj(hidden_states)  # (B, T, feature_dim*H)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)  # (B, T, v_dim)

        # (B, T, H, feature_dim)
        q = rearrange(q, "b t (h d) -> b t h d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b t h d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b t h d", h=self.num_heads)

        # ReBased feature map: (B, T, H, feature_dim) → (B, T, H, expanded_dim)
        q = self.feature_map_q(q)
        k = self.feature_map_k(k)

        # → (B, H, T, expanded_dim) for batch matmul
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Causal decay mask D: (H, T, T)
        D = torch.stack([
            _get_decay_mask(T, self.gammas[h].item(), hidden_states.device)
            for h in range(self.num_heads)
        ])

        # Retention: (Q @ K^T) * D @ V
        scale = self.expanded_dim ** -0.5
        attn = torch.einsum("bhtd,bhsd->bhts", q, k) * scale  # (B, H, T, T)
        attn = attn * D.unsqueeze(0)
        Y = torch.einsum("bhts,bhsd->bhtd", attn, v)           # (B, H, T, head_v_dim)

        return rearrange(Y, "b h t d -> b t (h d)")

    # ── recurrent mode ─────────────────────────────────────────────────────────
    def _recurrent_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        device = hidden_states.device

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = rearrange(q, "b t (h d) -> b t h d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b t h d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b t h d", h=self.num_heads)

        # Apply feature map to full sequence at once, then step through time
        q = self.feature_map_q(q)   # (B, T, H, expanded_dim)
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

        Y = torch.stack(outputs, dim=1)          # (B, T, H, head_v_dim)
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
