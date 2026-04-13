# -*- coding: utf-8 -*-
"""
GatedDeltaNet with ReBased feature map on Q/K.

Same as GatedDeltaNet but Q/K go through:
  1. Project hidden_size -> feature_dim*num_heads
  2. RebasedFeatureMap: LayerNorm + upper-triangular outer product
     output_dim per head = feature_dim*(feature_dim+1)//2

State size: num_heads x expanded_dim x head_v_dim
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
    from fla.ops.gated_delta_rule import (chunk_gated_delta_rule,
                                          fused_recurrent_gated_delta_rule)
except Exception:
    raise ImportError("Need to install fla: pip install flash-linear-attention")

from fla.modules.feature_map import RebasedFeatureMap
from zoology.mixers.feature_maps.rebased import rebased_expanded_size


class GatedDeltaNetReBased(nn.Module):
    """
    GatedDeltaNet with ReBased quadratic feature map on Q/K.

    Args:
        d_model:        model (hidden) dimension
        feature_dim:    per-head input dim to the feature map;
                        expanded key dim = feature_dim*(feature_dim+1)//2
        expand_v:       value expansion ratio
        num_heads:      number of attention heads
        use_gate:       output gate
        use_short_conv: short causal convolution on Q/K/V
        conv_size:      kernel size for short conv
        layer_idx:      layer index
        use_gamma / use_beta_fm / normalize:
                        hyperparams of RebasedFeatureMap
    """

    def __init__(
        self,
        d_model: int = 2048,
        feature_dim: int = 11,
        expand_v: float = 1.0,
        num_heads: int = 4,
        mode: str = 'chunk',
        use_gate: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        use_gamma: bool = True,
        use_beta_fm: bool = True,
        normalize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        assert mode in ['chunk', 'fused_recurrent']
        self.mode = mode
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.expanded_dim = rebased_expanded_size(feature_dim)
        self.hidden_size = d_model
        self.head_dim = d_model // num_heads
        self.value_dim = int(self.head_dim * num_heads * expand_v)
        self.head_v_dim = self.value_dim // num_heads
        self.head_k_dim = self.expanded_dim
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.layer_idx = layer_idx
        self.silu = nn.SiLU()

        qk_proj_dim = feature_dim * num_heads

        self.q_proj = nn.Linear(d_model, qk_proj_dim, bias=False)
        self.k_proj = nn.Linear(d_model, qk_proj_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.b_proj = nn.Linear(d_model, num_heads, bias=False)
        self.a_proj = nn.Linear(d_model, num_heads, bias=False)

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

        # Gating params (same as GatedDeltaNet)
        A = torch.empty(num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(num_heads))
        self.D._no_weight_decay = True
        dt_min, dt_max, dt_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            torch.rand(num_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.q_conv1d = ShortConvolution(qk_proj_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(qk_proj_dim, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')

        if use_gate:
            self.g_proj = nn.Linear(d_model, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:

        mode = self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # ── Q / K / V ──────────────────────────────────────────────────────────
        if self.use_short_conv:
            conv_state_q = conv_state_k = conv_state_v = None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            pos_ids = kwargs.get('position_ids', None)
            q, conv_state_q = self.q_conv1d(self.q_proj(hidden_states), mask=conv_mask,
                                             cache=conv_state_q, output_final_state=use_cache,
                                             seq_idx=pos_ids)
            k, conv_state_k = self.k_conv1d(self.k_proj(hidden_states), mask=conv_mask,
                                             cache=conv_state_k, output_final_state=use_cache,
                                             seq_idx=pos_ids)
            v, conv_state_v = self.v_conv1d(self.v_proj(hidden_states), mask=conv_mask,
                                             cache=conv_state_v, output_final_state=use_cache,
                                             seq_idx=pos_ids)
        else:
            q = self.silu(self.q_proj(hidden_states))
            k = self.silu(self.k_proj(hidden_states))
            v = self.silu(self.v_proj(hidden_states))

        # (B, T, H, feature_dim)
        q = rearrange(q, 'b t (h d) -> b t h d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b t h d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.num_heads)

        # ReBased feature map: (B, T, H, feature_dim) -> (B, T, H, expanded_dim)
        q = self.feature_map_q(q)
        k = self.feature_map_k(k)

        # L2-normalize after feature map — same reason as DeltaNetReBased.
        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        k = torch.nn.functional.normalize(k, p=2, dim=-1)

        beta = self.b_proj(hidden_states).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(
            self.a_proj(hidden_states).float() + self.dt_bias
        )

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
                use_qk_l2norm_in_kernel=False,
            )
        else:
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=False,
            )

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[1],
            )

        if self.use_gate:
            g_out = rearrange(self.g_proj(hidden_states), 'b t (h d) -> b t h d', h=self.num_heads)
            o = self.o_norm(o, g_out)
        else:
            o = self.o_norm(o)

        o = rearrange(o, 'b t h d -> b t (h d)').to(torch.float32)
        return self.o_proj(o)

    def state_size(self, **kwargs) -> int:
        return self.num_heads * self.expanded_dim * self.head_v_dim
