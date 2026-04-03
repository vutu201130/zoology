"""
DeltaNet baseline only - for fair comparison with kernel variants.

Run:
    python -m zoology.launch zoology/experiments/kernel_deltanet/baseline_only.py
"""
from zoology.experiments.kernel_deltanet.run import configs

configs = [c for c in configs if "baseline" in c.run_id]
