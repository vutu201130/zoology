"""
Baseline experiment: softmax attention vs linear attention approximations.

Goal: show clearly WHERE and HOW MUCH softmax outperforms linear attention.

Key design choices to maximize the gap:
  1. num_kv_pairs up to 64 in TRAINING (hard enough that state compression matters)
  2. num_kv_pairs up to 128 in TEST (OOD extrapolation)
  3. vocab_size=8192 (keys are distinct, not trivially separable)
  4. d_model=128 only — small state → linear methods more overloaded
  5. Models: attention (upper bound) vs based, delta_net, gated_delta_net, gla

State size analysis (d_model=128, num_heads=2, head_dim=64):
  - DeltaNet / GatedDeltaNet: 2 × 64 × 64 = 8192 floats per layer
  - Based (ftr_dim=8): 2 × 73 × 64 = 9344 floats per layer (73 = 8²+8+1)
  - Attention: grows with seq_len (O(N)), no fixed limit

Gap is largest when num_kv_pairs > state_capacity, which we hit at num_kv_pairs=32~64.

Run:
    python -m zoology.launch zoology/experiments/kernel_deltanet/run_baseline_gap.py -p
"""

import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.experiments.models_repo import (
    add_attention, add_based, add_delta_net, add_gated_delta_net, add_gla
)

sweep_id = uuid.uuid4().hex[:6]
sweep_name = "baseline_gap_" + sweep_id

VOCAB_SIZE = 8_192
PREDICTIONS_DIR = "/scratch/mb26/tv8394/workspace/zoology/predictions"

# ── Data: hard settings where state compression becomes a bottleneck ───────────
# Training: escalate difficulty, cap at num_kv_pairs=64
train_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,  num_examples=50_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=20_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512, num_examples=20_000, num_kv_pairs=64),
]

# Test: go OOD to show extrapolation gap
test_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,  num_examples=1_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,  num_examples=1_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=1_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=1_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512, num_examples=1_000, num_kv_pairs=64),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512, num_examples=1_000, num_kv_pairs=128),
]

input_seq_len = max(c.input_seq_len for c in train_configs + test_configs)
batch_size = 128

data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    batch_size=(batch_size, batch_size // 4),
    cache_dir="/scratch/mb26/tv8394/workspace/zoology/data_baseline_gap",
)

# ── Model setup ───────────────────────────────────────────────────────────────
conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={"l_max": input_seq_len, "kernel_size": 3, "implicit_long_conv": True},
)

model_factory_kwargs = {
    "state_mixer": dict(name="torch.nn.Identity", kwargs={}),
    "vocab_size": VOCAB_SIZE,
}

# Only d_model=128: small enough state to show compression bottleneck clearly
models = []
models = add_attention(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_based(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_gated_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_gla(models, conv_mixer, input_seq_len, model_factory_kwargs)

# Filter to d_model=128 only — this is the "show the gap" experiment
models = [m for m in models if m.d_model == 128]

# ── Sweep ──────────────────────────────────────────────────────────────────────
configs = []
for model in models:
    for lr in np.logspace(-3, -1.5, 4):
        run_id = f"find_gap_{model.name}-d{model.d_model}-lr{lr:.1e}"
        configs.append(TrainConfig(
            model=model,
            data=data,
            learning_rate=lr,
            max_epochs=32,
            logger=LoggerConfig(
                project_name="zoology",
                entity="vutu201130-matrixone",
            ),
            slice_keys=["num_kv_pairs"],
            sweep_id=sweep_name,
            run_id=run_id,
            predictions_path=f"{PREDICTIONS_DIR}/{run_id}",
            collect_predictions=True,
        ))

print(f">> Total configs: {len(configs)}")
for c in configs:
    print(f"   {c.run_id}")
