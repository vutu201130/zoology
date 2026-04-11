"""
ReBased feature map — wraps fla.modules.feature_map.RebasedFeatureMap.

Reference: https://arxiv.org/abs/2309.12307
Implementation: https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/feature_map.py

How it works:
  1. LayerNorm with learnable gamma/beta
  2. Quadratic outer product x⊗x, upper-triangular only:
       x2_1 = diagonal  (shape: ..., d)         -> x_i²
       x2_2 = off-diag  (shape: ..., d*(d-1)/2) -> x_i*x_j  (i<j)
  3. Concatenate with separate scalings:
       output = [x2_2 * d^(-0.5),  x2_1 * (2/d)^0.5]
       output_dim = d*(d-1)/2 + d = d*(d+1)/2

So: feature_dim=8  -> expanded = 36
    feature_dim=11 -> expanded = 66  (≈ 64, same state as default head_k_dim=64)
"""

from fla.modules.feature_map import RebasedFeatureMap  # noqa: F401 — re-export


def rebased_expanded_size(feature_dim: int) -> int:
    """Output dimension of RebasedFeatureMap given input feature_dim."""
    return feature_dim * (feature_dim + 1) // 2
