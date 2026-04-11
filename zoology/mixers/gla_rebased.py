# -*- coding: utf-8 -*-
"""
GLA with ReBased feature map on Q/K.

For GLA, Q/K already have shape (B, T, H, head_k_dim).
RebasedFeatureMap takes head_k_dim -> head_k_dim*(head_k_dim+1)//2.

Since GLA's gk gate operates on the key dimension, expanding Q/K changes the
head_k_dim. We re-project Q and K to a smaller feature_dim first so that
after expansion the key dim stays comparable to the original head_k_dim.

Default: feature_dim=11 -> expanded=66 ≈ head_k_dim=64 (for d_model=128, num_heads=2)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.modules.feature_map import RebasedFeatureMap
from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla

from zoology.mixers.feature_maps.rebased import rebased_expanded_size


class GLAReBased(nn.Module):
    """
    GLA with ReBased quadratic feature map on Q/K.

    Args:
        d_model:        model (hidden) dimension
        feature_dim:    per-head input dim to feature map;
                        expanded key dim = feature_dim*(feature_dim+1)//2
        expand_v:       value expansion ratio (default 1.0)
        num_heads:      number of attention heads
        mode:           'chunk' | 'fused_recurrent' | 'fused_chunk'
        gate_logit_normalizer: divisor for logsigmoid gate (default 16)
        gate_low_rank_dim: low-rank dim for gate projection
        use_output_gate: use SwiGLU output gate
        use_gamma / use_beta_fm / normalize:
                        hyperparams of RebasedFeatureMap
        layer_idx:      layer index
    """

    def __init__(
        self,
        d_model: int = 1024,
        feature_dim: int = 11,
        expand_v: float = 1.0,
        num_heads: int = 4,
        mode: str = 'chunk',
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        use_output_gate: bool = True,
        gate_fn: str = 'swish',
        elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        fuse_norm: bool = True,
        clamp_min: Optional[float] = None,
        layer_idx: int = None,
        use_gamma: bool = True,
        use_beta_fm: bool = True,
        normalize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        assert mode in ['chunk', 'fused_recurrent', 'fused_chunk']
        self.mode = mode
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.expanded_dim = rebased_expanded_size(feature_dim)   # key dim after feature map
        self.hidden_size = d_model
        self.value_dim = int(d_model * expand_v)
        self.head_v_dim = self.value_dim // num_heads
        self.gate_logit_normalizer = gate_logit_normalizer
        self.clamp_min = clamp_min
        self.use_output_gate = use_output_gate
        self.layer_idx = layer_idx

        assert self.value_dim % num_heads == 0

        qk_proj_dim = feature_dim * num_heads

        # Q/K projected to feature_dim*num_heads (smaller than usual key_dim)
        self.q_proj = nn.Linear(d_model, qk_proj_dim, bias=False)
        self.k_proj = nn.Linear(d_model, qk_proj_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False)

        # Decay gate: operates on feature_dim per head (before expansion)
        self.gk_proj = nn.Sequential(
            nn.Linear(d_model, gate_low_rank_dim, bias=False),
            nn.Linear(gate_low_rank_dim, qk_proj_dim, bias=True),
        )

        if use_output_gate:
            self.g_proj = nn.Linear(d_model, self.value_dim, bias=False)

        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)

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

        if gate_fn == 'swish' and fuse_norm and use_output_gate:
            self.g_norm_swish_gate = FusedRMSNormSwishGate(self.head_v_dim, elementwise_affine, norm_eps)
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = RMSNorm(hidden_size=self.head_v_dim, elementwise_affine=elementwise_affine, eps=norm_eps)
            from fla.modules.activations import ACT2FN
            self.gate_fn = ACT2FN[gate_fn]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:

        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None

        # ── Q / K / V / gk ────────────────────────────────────────────────────
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        gk = self.gk_proj(hidden_states)   # decay gate, still in feature_dim space

        if attention_mask is not None:
            v = v.mul_(attention_mask[:, -v.shape[-2]:, None])

        # (B, T, H, feature_dim)
        q = rearrange(q, 'b t (h d) -> b t h d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b t h d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.num_heads)
        gk = rearrange(gk, 'b t (h d) -> b t h d', h=self.num_heads)

        # decay gate: logsigmoid on feature_dim, then expand to expanded_dim
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer  # (B, T, H, feature_dim)
        # repeat each gate value for the expanded dims it corresponds to
        # Simple approach: tile gk to match expanded_dim
        # feature_dim=11 -> 66 dims: we tile and truncate
        repeats = (self.expanded_dim + self.feature_dim - 1) // self.feature_dim
        gk = gk.repeat(1, 1, 1, repeats)[..., :self.expanded_dim]

        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)

        # Apply ReBased feature map: (B, T, H, feature_dim) -> (B, T, H, expanded_dim)
        q = self.feature_map_q(q)
        k = self.feature_map_k(k)

        cu_seqlens = kwargs.get('cu_seqlens', None)

        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(
                q=q, k=k, v=v, gk=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        elif mode == 'fused_chunk':
            o, recurrent_state = fused_chunk_gla(
                q=q, k=k, v=v, g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
            )
        else:
            o, recurrent_state = chunk_gla(
                q=q, k=k, v=v, g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=None,
                layer_idx=self.layer_idx,
                offset=q.shape[1],
            )

        if self.use_output_gate:
            g = self.g_proj(hidden_states)
            if self.fuse_norm_and_gate:
                g = rearrange(g, 'b t (h d) -> b t h d', h=self.num_heads)
                o = self.g_norm_swish_gate(o, g)
                o = rearrange(o, 'b t h d -> b t (h d)')
            else:
                o = rearrange(self.g_norm(o), 'b t h d -> b t (h d)')
                o = o * self.gate_fn(g)
        else:
            o = rearrange(self.g_norm(o), 'b t h d -> b t (h d)')

        return self.o_proj(o)

    def state_size(self, **kwargs) -> int:
        return self.num_heads * self.expanded_dim * self.head_v_dim
