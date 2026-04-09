"""
Experiment: Multi-Scale Position-Modulated DeltaNet vs baselines on hard MQAR.

Settings chosen to maximise the gap between softmax attention and linear attention:
  - num_kv_pairs up to 64 (train), tested up to 128 (OOD)
  - seq_len up to 512
  - vocab_size = 8192 (large, hard to memorise by accident)

Models compared:
  1. attention_baseline   — softmax MHA  (upper bound)
  2. delta_net_baseline   — DeltaNet, no gating
  3. gated_delta_net      — GatedDeltaNet (single decay, our direct competitor)
  4. delta_net_ms4        — DeltaNetMultiScale, n_scales=4
  5. delta_net_ms8        — DeltaNetMultiScale, n_scales=8

Run:
    python -m zoology.launch zoology/experiments/kernel_deltanet/run_multiscale.py -p
"""

import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

sweep_id = uuid.uuid4().hex[:6]
sweep_name = "multiscale_" + sweep_id

VOCAB_SIZE = 8_192
D_MODEL = 128    # single d_model to keep sweep tractable

# ── Data: hard settings where attention >> linear attention ────────────────────
train_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,  num_examples=20_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=20_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512, num_examples=20_000, num_kv_pairs=64),
]
test_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,   num_examples=1_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128,  num_examples=1_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256,  num_examples=1_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256,  num_examples=1_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512,  num_examples=1_000, num_kv_pairs=64),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512,  num_examples=1_000, num_kv_pairs=128),
]

input_seq_len = max(c.input_seq_len for c in train_configs + test_configs)

import os
cache_dir = os.path.join(os.path.dirname(__file__), "../../../../data_multiscale")

data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    batch_size=(32, 32),
    cache_dir=os.path.abspath(cache_dir),
)

# ── Shared model kwargs ────────────────────────────────────────────────────────
conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={"l_max": input_seq_len, "kernel_size": 3, "implicit_long_conv": True},
)

base_model_kwargs = dict(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_layers=2,
    block_type="TransformerBlock",
    max_position_embeddings=0,
    state_mixer=dict(name="torch.nn.Identity", kwargs={}),
)

# ── Model definitions ──────────────────────────────────────────────────────────
models = []

# 1. Softmax attention — upper bound
models.append(ModelConfig(
    sequence_mixer=ModuleConfig(
        name="zoology.mixers.hybrid.Hybrid",
        kwargs={"configs": [conv_mixer, dict(
            name="zoology.mixers.attention.MHA",
            kwargs={"num_heads": 2}
        )]},
    ),
    name="attention_baseline",
    **base_model_kwargs,
))

# 2. DeltaNet — no gating (lower bound for gated variants)
models.append(ModelConfig(
    sequence_mixer=ModuleConfig(
        name="zoology.mixers.hybrid.Hybrid",
        kwargs={"configs": [conv_mixer, dict(
            name="zoology.mixers.delta_net.DeltaNet",
            kwargs={"l_max": input_seq_len, "num_heads": 2,
                    "use_beta": True, "use_gate": False,
                    "use_short_conv": True, "conv_size": 4}
        )]},
    ),
    name="delta_net_baseline",
    **base_model_kwargs,
))

# 3. GatedDeltaNet — single decay gate (direct competitor)
models.append(ModelConfig(
    sequence_mixer=ModuleConfig(
        name="zoology.mixers.hybrid.Hybrid",
        kwargs={"configs": [conv_mixer, dict(
            name="zoology.mixers.gated_delta_net.GatedDeltaNet",
            kwargs={"num_heads": 2, "expand_v": 1,
                    "use_gate": False, "use_short_conv": True, "conv_size": 4}
        )]},
    ),
    name="gated_delta_net",
    **base_model_kwargs,
))

# 4. DeltaNetMultiScale n_scales=4 — our method
models.append(ModelConfig(
    sequence_mixer=ModuleConfig(
        name="zoology.mixers.hybrid.Hybrid",
        kwargs={"configs": [conv_mixer, dict(
            name="zoology.mixers.delta_net_multiscale.DeltaNetMultiScale",
            kwargs={"num_heads": 2, "n_scales": 4, "expand_v": 1,
                    "use_gate": False, "use_short_conv": True, "conv_size": 4}
        )]},
    ),
    name="delta_net_ms4",
    **base_model_kwargs,
))

# 5. DeltaNetMultiScale n_scales=8 — more expressive
models.append(ModelConfig(
    sequence_mixer=ModuleConfig(
        name="zoology.mixers.hybrid.Hybrid",
        kwargs={"configs": [conv_mixer, dict(
            name="zoology.mixers.delta_net_multiscale.DeltaNetMultiScale",
            kwargs={"num_heads": 2, "n_scales": 8, "expand_v": 1,
                    "use_gate": False, "use_short_conv": True, "conv_size": 4}
        )]},
    ),
    name="delta_net_ms8",
    **base_model_kwargs,
))

# ── Sweep over LRs ─────────────────────────────────────────────────────────────
configs = []
for model in models:
    for lr in [3e-4, 1e-3, 3e-3]:
        run_id = f"{model.name}-d{model.d_model}-lr{lr:.1e}"
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
        ))

print(f">> Total configs: {len(configs)}")
for c in configs:
    print(f"   {c.run_id}")
