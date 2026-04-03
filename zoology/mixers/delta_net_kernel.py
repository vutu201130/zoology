# -*- coding: utf-8 -*-
"""
DeltaNet with learnable kernel feature maps for Q and K.

Idea: apply a feature map φ to Q and K before the delta rule update,
so that the similarity between keys is computed in a richer feature space.
This may improve recall by making keys more separable.

Usage in config:
    ModuleConfig(
        name="zoology.mixers.delta_net_kernel.DeltaNetKernel",
        kwargs={
            "l_max": input_seq_len,
            "num_heads": 2,
            "use_beta": True,
            "use_gate": False,
            "use_short_conv": True,
            "conv_size": 4,
            "feature_map": "taylor_exp",  # or "identity", "pos_elu", "rebased"
            "feature_dim": 8,             # dim BEFORE expansion (keep small!)
        }
    )
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

try:
    from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
    from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule
except:
    assert 0, print("Need to install fla: pip install flash-linear-attention")

from zoology.mixers.based import init_feature_map

if TYPE_CHECKING:
    from fla.models.utils import Cache


def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)


def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


class DeltaNetKernel(nn.Module):
    """
    DeltaNet + feature map φ applied to Q and K.

    Key difference from DeltaNet:
    - q_proj and k_proj project to feature_dim (small) instead of key_dim
    - feature map φ expands feature_dim → expanded_dim
    - delta rule operates in expanded space
    - v_proj remains the same

    Args:
        feature_map (str): which feature map to use. Options:
            "identity"  - no change, equivalent to original DeltaNet
            "taylor_exp" - 2nd-order Taylor approx of exp (Based kernel)
            "pos_elu"   - ELU + 1
        feature_dim (int): input dim to feature map BEFORE expansion.
            Keep small (e.g. 8, 16) since expanded dim = feature_dim^2 + feature_dim + 1.
    """

    def __init__(
        self,
        mode: str = 'chunk',
        d_model: int = None,
        expand_v: float = 1.0,
        num_heads: int = 4,
        use_beta: bool = True,
        use_gate: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        feature_map: str = 'taylor_exp',
        feature_dim: int = 8,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
        **kwargs
    ) -> DeltaNetKernel:
        super().__init__()

        self.mode = mode
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.layer_idx = layer_idx
        self.feature_dim = feature_dim

        hidden_size = int(d_model)
        self.hidden_size = hidden_size
        self.value_dim = int(hidden_size * expand_v)
        self.head_v_dim = self.value_dim // num_heads

        # feature map: maps feature_dim → expanded_dim
        self.feature_map = init_feature_map(
            feature_map=feature_map,
            input_dim=feature_dim,
            head_dim_idx=-1,
            eps=1e-12,
        )
        self.expanded_dim = int(self.feature_map.expanded_size())

        # Q, K project to feature_dim (small), then expanded by φ
        self.q_proj = nn.Linear(hidden_size, feature_dim * num_heads, bias=False)
        self.k_proj = nn.Linear(hidden_size, feature_dim * num_heads, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=feature_dim * num_heads,
                kernel_size=conv_size,
                activation='silu'
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=feature_dim * num_heads,
                kernel_size=conv_size,
                activation='silu'
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                activation='silu'
            )

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        # output projects from expanded_dim (key space) back to hidden
        # but delta rule output is in value space, so proj from value_dim
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        **kwargs
    ) -> torch.Tensor:

        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode

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
                cache=conv_state_q, output_final_state=use_cache
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states), mask=conv_mask,
                cache=conv_state_k, output_final_state=use_cache
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states), mask=conv_mask,
                cache=conv_state_v, output_final_state=use_cache
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        # reshape to (b, l, h, feature_dim)
        q = rearrange(q, '... (h d) -> ... h d', d=self.feature_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.feature_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        # apply feature map φ: (b, l, h, feature_dim) → (b, l, h, expanded_dim)
        q = self.feature_map(q)
        k = self.feature_map(k)

        # l2 normalize in expanded space
        q = F.normalize(q, dim=-1).to(torch.bfloat16)
        k = F.normalize(k, dim=-1).to(torch.bfloat16)
        v = v.to(torch.bfloat16)

        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = q.new_ones(q.shape[0], q.shape[1], self.num_heads)

        if attention_mask is not None:
            beta = beta.mul(attention_mask[:, -beta.shape[-2]:, None])

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        cu_seqlens = kwargs.get('cu_seqlens', None)

        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_delta_rule(
                q=q, k=k, v=v,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        elif mode == 'chunk':
            o, recurrent_state = chunk_delta_rule(
                q=q, k=k, v=v,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        o = o.float()

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)

        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        return o

    def state_size(self, sequence_length: int = 2048):
        # hidden state S: (num_heads, expanded_dim, head_v_dim)
        return self.num_heads * self.expanded_dim * self.head_v_dim
