"""
Multi-Scale DeltaNet vs baselines on hard MQAR.

Data settings mirror original_mqar_configs.py for apples-to-apples comparison.
Test configs go up to num_kv_pairs=256, seq_len=1024 (OOD extrapolation) to
make the gap between softmax attention and recurrent models clearly visible.

Models:
  - attention          (upper bound, from models_repo)
  - gated_delta_net    (single decay, direct competitor, from models_repo)
  - delta_net_ms4      (ours: 4-scale mixture decay)
  - delta_net_ms8      (ours: 8-scale mixture decay)

Run:
    python -m zoology.launch zoology/experiments/kernel_deltanet/run_multiscale.py -p
"""

import uuid
import numpy as np

from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.experiments.models_repo import add_attention, add_gated_delta_net, add_delta_net

sweep_id = uuid.uuid4().hex[:6]
sweep_name = "multiscale_mqar_" + sweep_id

VOCAB_SIZE = 8_192
PREDICTIONS_DIR = "/scratch/mb26/tv8394/workspace/zoology/predictions"

# ── Data: mirrors original_mqar_configs.py ────────────────────────────────────
train_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,  num_examples=100_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=20_000,  num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000,  num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000,  num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000,  num_kv_pairs=64),
]
test_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,   num_examples=1_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,   num_examples=1_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,   num_examples=1_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128,  num_examples=1_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256,  num_examples=1_000, num_kv_pairs=64),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512,  num_examples=1_000, num_kv_pairs=128),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024, num_examples=1_000, num_kv_pairs=256),
]

input_seq_len = max(c.input_seq_len for c in train_configs + test_configs)
batch_size = 256

data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    batch_size=(batch_size, batch_size // 8),
    cache_dir="/scratch/mb26/tv8394/workspace/zoology/data_multiscale",
)

# ── Shared model kwargs ────────────────────────────────────────────────────────
conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={"l_max": input_seq_len, "kernel_size": 3, "implicit_long_conv": True},
)

model_factory_kwargs = {
    "state_mixer": dict(name="torch.nn.Identity", kwargs={}),
    "vocab_size": VOCAB_SIZE,
}

# ── Baseline models from models_repo ──────────────────────────────────────────
models = []
models = add_attention(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_gated_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs)

# Keep only d_model=64 and 128 from repo models to reduce sweep size
models = [m for m in models if m.d_model in [64, 128]]

# ── Our models: DeltaNetMultiScale ────────────────────────────────────────────
for d_model in [64, 128]:
    for n_scales in [4, 8]:
        mixer = ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": [conv_mixer, dict(
                name="zoology.mixers.delta_net_multiscale.DeltaNetMultiScale",
                kwargs={
                    "num_heads": 2,
                    "n_scales": n_scales,
                    "expand_v": 1,
                    "use_gate": False,
                    "use_short_conv": True,
                    "conv_size": 4,
                }
            )]},
        )
        models.append(ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=2,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name=f"delta_net_ms{n_scales}",
            **model_factory_kwargs,
        ))

# ── Sweep ──────────────────────────────────────────────────────────────────────
configs = []
for model in models:
    for lr in np.logspace(-3, -1.5, 4):
        run_id = f"gadi_{model.name}-d{model.d_model}-lr{lr:.1e}"
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
