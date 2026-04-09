# -*- coding: utf-8 -*-
"""
DeltaNet with Multi-Scale Position-Modulated Kernel Decay.

Motivation (Proposal 3, kernel theory framing):
    GatedDeltaNet uses a single input-dependent decay:
        γ_t = exp(-A * softplus(a(x_t) + dt_bias))   ← one timescale per head

    This corresponds to a stationary kernel:
        f(τ) = exp(-λ|τ|)  ← single exponential, one timescale

    By Bochner's theorem, any mixture of positive definite kernels is PD:
        f(τ) = Σ_m w_m(context) · exp(-λ_m|τ|)

    → Richer kernel: model can pick slow (large τ) or fast (small τ) timescales
      depending on what the current context requires.

    Key distinction from GatedDeltaNet:
    - GatedDeltaNet: γ_t = single gate, fully input-dependent
    - DeltaNetMultiScale: γ_t = Σ_m w_m(x_t) · σ(s_m)
        where s_m are SHARED learnable timescale anchors
        and w_m(x_t) are input-dependent mixing weights
    → Separates "which timescales exist" (shared, learned) from
      "how much of each timescale" (per-token, input-dependent)

Usage:
    ModuleConfig(
        name="zoology.mixers.delta_net_multiscale.DeltaNetMultiScale",
        kwargs={
            "num_heads": 4,
            "n_scales": 4,       # number of decay timescales
            "use_gate": False,
            "use_short_conv": True,
            "conv_size": 4,
        }
    )
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

try:
    from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
    from fla.ops.gated_delta_rule import (chunk_gated_delta_rule,
                                          fused_recurrent_gated_delta_rule)
except Exception:
    assert False, "Need to install fla: pip install flash-linear-attention"

if TYPE_CHECKING:
    from fla.models.utils import Cache


class DeltaNetMultiScale(nn.Module):
    """
    DeltaNet with multi-scale mixture decay.

    Args:
        d_model (int): Hidden size.
        num_heads (int): Number of attention heads.
        n_scales (int): Number of decay timescales in the mixture.
            Each scale has a learnable base decay rate σ(s_m) ∈ (0,1).
            Input-dependent mixing weights w_m(x_t) select how much of each scale to use.
        expand_v (float): Value dimension multiplier. Default 1.0.
        use_gate (bool): Output gate (SwiGLU-style). Default False.
        use_short_conv (bool): Short conv before QKV. Default True.
        conv_size (int): Short conv kernel size. Default 4.
    """

    def __init__(
        self,
        d_model: int = None,
        num_heads: int = 4,
        n_scales: int = 4,
        expand_v: float = 1.0,
        mode: str = 'chunk',
        use_gate: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
        **kwargs,
    ) -> DeltaNetMultiScale:
        super().__init__()

        self.mode = mode
        self.num_heads = num_heads
        self.n_scales = n_scales
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.layer_idx = layer_idx

        hidden_size = int(d_model)
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.key_dim = num_heads * self.head_dim
        self.value_dim = int(self.key_dim * expand_v)
        self.head_v_dim = self.value_dim // num_heads

        # QKV projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)  # beta (write strength)

        # ── Multi-scale decay ──────────────────────────────────────────────────
        # Learnable timescale anchors: s_m → decay = sigmoid(s_m) ∈ (0,1)
        # Initialize to cover a range of timescales:
        #   sigmoid(0.0)  ≈ 0.50  (fast: forget half each step)
        #   sigmoid(2.2)  ≈ 0.90
        #   sigmoid(4.6)  ≈ 0.99  (slow: remember for ~100 steps)
        init_vals = torch.linspace(0.0, 4.6, n_scales)  # (M,)
        # Repeat for each head: (H, M)
        self.log_decay_scales = nn.Parameter(
            init_vals.unsqueeze(0).expand(num_heads, -1).clone()
        )
        self.log_decay_scales._no_weight_decay = True

        # Input-dependent mixing weights: x_t → w ∈ Δ^{M-1} per head
        self.mix_proj = nn.Linear(hidden_size, num_heads * n_scales, bias=True)
        # Zero-init so model starts with uniform mixing (equal weight to all scales)
        nn.init.zeros_(self.mix_proj.weight)
        nn.init.zeros_(self.mix_proj.bias)
        # ──────────────────────────────────────────────────────────────────────

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim, kernel_size=conv_size, activation='silu'
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim, kernel_size=conv_size, activation='silu'
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim, kernel_size=conv_size, activation='silu'
            )

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _compute_decay(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale mixture decay gate.

        Returns g of shape (B, T, H) with g < 0, matching fla's expected format
        for chunk_gated_delta_rule (which internally computes exp(g) as state decay).

        γ_t = Σ_m w_m(x_t) · σ(s_m)   ∈ (0, 1)
        g_t = log(γ_t)                  ∈ (-∞, 0)
        """
        B, T, _ = hidden_states.shape

        # Base decay rates per scale (shared across time): (H, M)
        decay_m = torch.sigmoid(self.log_decay_scales)  # (H, M), in (0,1)

        # Input-dependent mixing weights: (B, T, H, M)
        mix_logits = self.mix_proj(hidden_states)           # (B, T, H*M)
        mix_logits = rearrange(mix_logits, 'b t (h m) -> b t h m', h=self.num_heads)
        w = F.softmax(mix_logits, dim=-1)                   # (B, T, H, M), sums to 1

        # Mixture decay: γ_t = Σ_m w_m · decay_m
        gamma = (w * decay_m[None, None, :, :]).sum(dim=-1)  # (B, T, H), in (0,1)

        # Convert to log-space for fla (g must be < 0)
        g = gamma.clamp(min=1e-6).log()  # (B, T, H), in (-∞, 0)
        return g

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:

        mode = self.mode if self.training else ('fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode)

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None

        if self.use_short_conv:
            conv_state_q = conv_state_k = conv_state_v = None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states), mask=conv_mask,
                cache=conv_state_q, output_final_state=use_cache,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states), mask=conv_mask,
                cache=conv_state_k, output_final_state=use_cache,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states), mask=conv_mask,
                cache=conv_state_v, output_final_state=use_cache,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        q = rearrange(q, 'b t (h d) -> b t h d', d=self.head_dim)
        k = rearrange(k, 'b t (h d) -> b t h d', d=self.head_dim)
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)

        beta = self.b_proj(hidden_states).sigmoid()      # (B, T, H)
        g = self._compute_decay(hidden_states)           # (B, T, H), log-space

        if attention_mask is not None:
            beta = beta.mul(attention_mask[:, -beta.shape[-2]:, None])
            g = g.mul(attention_mask[:, -g.shape[-2]:, None])

        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        beta = beta.to(torch.bfloat16)
        g = g.to(torch.bfloat16)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        cu_seqlens = kwargs.get('cu_seqlens', None)

        if mode == 'chunk':
            o, recurrent_state = chunk_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        elif mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"mode `{mode}` not supported.")

        o = o.float()

        if self.use_gate:
            gate = rearrange(self.g_proj(hidden_states), 'b t (h d) -> b t h d', d=self.head_v_dim)
            o = self.o_norm(o, gate)
        else:
            o = self.o_norm(o)

        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        return o

    def state_size(self, sequence_length: int = 2048):
        return self.num_heads * self.head_dim * self.head_v_dim
