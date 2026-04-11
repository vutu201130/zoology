# -*- coding: utf-8 -*-
"""
RetNet mixer for Zoology.

Adapted from "Retentive Network: A Successor to Transformer for Large Language Models"
https://arxiv.org/pdf/2307.08621.pdf
Original repo: https://github.com/Jamie-Stirling/RetNet

Key mechanism:
  - Multi-scale exponential decay: γ_h ∈ [1-1/32, 1-1/512] per head
  - XPOS rotary position embedding on Q and K
  - Parallel mode (training): ret = (Q @ K^T) * D @ V
    where D[i,j] = γ^(i-j) if i≥j else 0  (causal decay mask)
  - Recurrent mode (inference): S_n = γ * S_{n-1} + K_n^T @ V_n
  - GroupNorm + swish output gate

API: forward(hidden_states, **kwargs) -> tensor  (same as other zoology mixers)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ── XPOS (xPos rotary position embedding) ─────────────────────────────────────

def _fixed_pos_embedding(x: torch.Tensor):
    """x: (seq_len, dim) → (sin, cos) both (seq_len, dim)"""
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, device=x.device) / dim))
    t = torch.arange(0, seq_len, dtype=torch.float, device=x.device)
    sinusoid = torch.einsum("i,j->ij", t, inv_freq)
    return torch.sin(sinusoid), torch.cos(sinusoid)


def _rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def _duplicate_interleave(m: torch.Tensor) -> torch.Tensor:
    dim0 = m.shape[0]
    m = m.view(-1, 1).repeat(1, 2).view(dim0, -1)
    return m


def _apply_rotary_pos_emb(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor,
                          scale: torch.Tensor) -> torch.Tensor:
    sin = _duplicate_interleave(sin * scale)
    cos = _duplicate_interleave(cos * scale)
    return (x * cos[:, :x.shape[-1]]) + (_rotate_every_two(x) * sin)[:, :, :x.shape[-1]]


class XPOS(nn.Module):
    def __init__(self, head_dim: int, scale_base: int = 512):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale",
            (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, x: torch.Tensor, offset: int = 0, downscale: bool = False):
        """x: (batch, seq_len, dim)"""
        length = x.shape[1]
        max_pos = length + offset
        scale = self.scale ** torch.arange(offset, max_pos, device=x.device).float().div(self.scale_base)[:, None]
        sin, cos = _fixed_pos_embedding(scale)
        if scale.shape[0] > length:
            scale, sin, cos = scale[-length:], sin[-length:], cos[-length:]
        if downscale:
            scale = 1.0 / scale
        return _apply_rotary_pos_emb(x, sin, cos, scale)


# ── Causal decay mask ─────────────────────────────────────────────────────────

def _get_decay_mask(seq_len: int, gamma: float, device: torch.device) -> torch.Tensor:
    """D[i,j] = gamma^(i-j) if i>=j else 0  shape: (seq_len, seq_len)"""
    n = torch.arange(seq_len, device=device).unsqueeze(1)
    m = torch.arange(seq_len, device=device).unsqueeze(0)
    D = (gamma ** (n - m)) * (n >= m).float()
    D = torch.nan_to_num(D, nan=0.0)
    return D


# ── RetNet mixer ──────────────────────────────────────────────────────────────

class RetNet(nn.Module):
    """
    Multi-scale Retention mixer, compatible with Zoology's TransformerBlock.

    Args:
        d_model:     hidden (model) dimension
        num_heads:   number of retention heads; each gets a different γ
        double_v_dim: double the V dimension (uses more params, not recommended for fair compare)
        recurrent_threshold: use recurrent mode when seq_len <= this (for inference)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 2,
        double_v_dim: bool = False,
        recurrent_threshold: int = 64,
        **kwargs,
    ) -> None:
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.v_dim = d_model * 2 if double_v_dim else d_model
        self.head_v_dim = self.v_dim // num_heads
        self.recurrent_threshold = recurrent_threshold

        # γ: linearly spaced in log-space, one per head
        # range: 1 - exp(log(1/32)) to 1 - exp(log(1/512)) ≈ 0.969 to 0.998
        gammas = 1 - torch.exp(
            torch.linspace(math.log(1 / 32), math.log(1 / 512), num_heads)
        )
        self.register_buffer("gammas", gammas)  # (H,)

        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, self.v_dim, bias=False)

        # Output gate (swish) + output projection
        self.g_proj = nn.Linear(d_model, self.v_dim, bias=False)
        self.o_proj = nn.Linear(self.v_dim, d_model, bias=False)

        # Group norm across heads (applied after retention, before gate)
        self.group_norm = nn.GroupNorm(num_heads, self.v_dim)

        # XPOS per head
        self.xpos = XPOS(self.head_dim)

    # ── parallel mode (training) ───────────────────────────────────────────────
    def _parallel_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape

        Q = self.q_proj(hidden_states)  # (B, T, d_model)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)  # (B, T, v_dim)

        # reshape to (B, T, H, head_dim)
        Q = rearrange(Q, "b t (h d) -> b h t d", h=self.num_heads)
        K = rearrange(K, "b t (h d) -> b h t d", h=self.num_heads)
        V = rearrange(V, "b t (h d) -> b h t d", h=self.num_heads)

        # Apply XPOS per head — operate on (B*H, T, head_dim)
        BH = B * self.num_heads
        Q = self.xpos(Q.reshape(BH, T, self.head_dim)).reshape(B, self.num_heads, T, self.head_dim)
        K = self.xpos(K.reshape(BH, T, self.head_dim), downscale=True).reshape(B, self.num_heads, T, self.head_dim)

        # Causal decay mask D: (H, T, T)
        D = torch.stack([
            _get_decay_mask(T, self.gammas[h].item(), hidden_states.device)
            for h in range(self.num_heads)
        ])  # (H, T, T)

        # Retention: (Q @ K^T) * D → (B, H, T, T)  then @ V → (B, H, T, head_v_dim)
        scale = self.head_dim ** -0.5
        attn = torch.einsum("bhtd,bhsd->bhts", Q, K) * scale  # (B, H, T, T)
        attn = attn * D.unsqueeze(0)                           # broadcast over batch
        Y = torch.einsum("bhts,bhsd->bhtd", attn, V)          # (B, H, T, head_v_dim)

        # merge heads: (B, T, v_dim)
        Y = rearrange(Y, "b h t d -> b t (h d)")
        return Y

    # ── recurrent mode (inference) ─────────────────────────────────────────────
    def _recurrent_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        device = hidden_states.device

        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)

        Q = rearrange(Q, "b t (h d) -> b h t d", h=self.num_heads)
        K = rearrange(K, "b t (h d) -> b h t d", h=self.num_heads)
        V = rearrange(V, "b t (h d) -> b h t d", h=self.num_heads)

        # XPOS
        BH = B * self.num_heads
        Q = self.xpos(Q.reshape(BH, T, self.head_dim)).reshape(B, self.num_heads, T, self.head_dim)
        K = self.xpos(K.reshape(BH, T, self.head_dim), downscale=True).reshape(B, self.num_heads, T, self.head_dim)

        # State: (B, H, head_dim, head_v_dim)
        S = torch.zeros(B, self.num_heads, self.head_dim, self.head_v_dim, device=device)
        outputs = []

        scale = self.head_dim ** -0.5
        for t in range(T):
            q_t = Q[:, :, t, :]       # (B, H, head_dim)
            k_t = K[:, :, t, :]       # (B, H, head_dim)
            v_t = V[:, :, t, :]       # (B, H, head_v_dim)

            # S_t = γ * S_{t-1} + k_t^T @ v_t
            kv = torch.einsum("bhd,bhe->bhde", k_t, v_t)  # (B, H, d, v)
            S = self.gammas.view(1, -1, 1, 1) * S + kv

            # y_t = q_t @ S_t
            y_t = torch.einsum("bhd,bhde->bhe", q_t, S) * scale  # (B, H, head_v_dim)
            outputs.append(y_t)

        Y = torch.stack(outputs, dim=2)  # (B, H, T, head_v_dim)
        Y = rearrange(Y, "b h t d -> b t (h d)")
        return Y

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        B, T, _ = hidden_states.shape

        if T <= self.recurrent_threshold:
            Y = self._recurrent_forward(hidden_states)
        else:
            Y = self._parallel_forward(hidden_states)

        # GroupNorm: expects (B, C, *) — reshape to (B, v_dim, T) then back
        Y = self.group_norm(Y.transpose(1, 2)).transpose(1, 2)

        # Swish gate + output projection
        G = F.silu(self.g_proj(hidden_states))  # (B, T, v_dim)
        return self.o_proj(G * Y)

    def state_size(self, **kwargs) -> int:
        return self.num_heads * self.head_dim * self.head_v_dim
