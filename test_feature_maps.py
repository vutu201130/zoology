"""
Quick test: RebasedFeatureMap vs TaylorExp

Run: python test_feature_maps.py
"""
import torch
from zoology.mixers.feature_maps.taylor import TaylorExp
from fla.modules.feature_map import RebasedFeatureMap

d = 8  # input dim per head
B, T, H = 2, 16, 2  # batch, seq_len, heads
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

x = torch.randn(B, T, H, d, device=device)

# ── TaylorExp ──────────────────────────────────────────────────────────────────
taylor = TaylorExp(input_dim=d)
# TaylorExp expects head_dim_idx=-1 (default), input shape (..., d)
out_taylor = taylor(x)
print(f"TaylorExp:  input {list(x.shape)} -> output {list(out_taylor.shape)}")
print(f"  expected output dim: 1 + {d} + {d}^2 = {1 + d + d**2}")

# ── RebasedFeatureMap ──────────────────────────────────────────────────────────
rebased = RebasedFeatureMap(head_dim=d, use_gamma=True, use_beta=True, normalize=True).to(device)
out_rebased = rebased(x)
print(f"\nRebasedFeatureMap: input {list(x.shape)} -> output {list(out_rebased.shape)}")
print(f"  expected output dim: {d}*({d}+1)//2 = {d*(d+1)//2}")

# ── Sanity: kernel approximation k(q,k) = φ(q)·φ(k) ──────────────────────────
print("\n── Kernel approximation check ────────────────────────────────")
q = torch.randn(B, T, H, d, device=device)
k = torch.randn(B, T, H, d, device=device)

# TaylorExp: should approximate exp(q·k/sqrt(d))
phi_q_t = taylor(q)
phi_k_t = taylor(k)
kernel_taylor = (phi_q_t * phi_k_t).sum(-1)  # (B, T, H)
true_exp      = torch.exp((q * k).sum(-1) / d**0.5)
print(f"TaylorExp  kernel vs exp(q·k/√d):  mean_abs_err = {(kernel_taylor - true_exp).abs().mean():.4f}")

# ReBased: φ(q)·φ(k)
phi_q_r = rebased(q)
phi_k_r = rebased(k)
kernel_rebased = (phi_q_r * phi_k_r).sum(-1)
print(f"ReBased    kernel vs exp(q·k/√d):  mean_abs_err = {(kernel_rebased - true_exp).abs().mean():.4f}")
print(f"  (ReBased is not trying to approximate exp — just shows expressiveness)")

# ── All-positive check (important for linear attention stability) ──────────────
print("\n── All-positive check ────────────────────────────────────────")
print(f"TaylorExp  min value: {out_taylor.min():.4f}  (can be negative!)")
print(f"ReBased    min value: {out_rebased.min():.4f}  (always >= 0 since x*x >= 0)")
