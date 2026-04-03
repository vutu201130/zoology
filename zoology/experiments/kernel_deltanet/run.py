"""
Compare DeltaNet (baseline) vs DeltaNetKernel (with feature maps) on MQAR.

Run:
    python -m zoology.launch zoology/experiments/kernel_deltanet/run.py
"""
import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

sweep_id = uuid.uuid4().hex[:6]
sweep_name = "kernel_deltanet_" + sweep_id

VOCAB_SIZE = 8_192

# ── Data ──────────────────────────────────────────────────────────────────────
# Vary num_kv_pairs to stress-test recall capacity
train_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,  num_examples=50_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=20_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=64),
]
test_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,   num_examples=1_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128,  num_examples=1_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256,  num_examples=1_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256,  num_examples=1_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512,  num_examples=1_000, num_kv_pairs=64),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024, num_examples=1_000, num_kv_pairs=128),
]

input_seq_len = max(c.input_seq_len for c in train_configs + test_configs)
data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    batch_size=(64, 64),
    cache_dir="/scratch/mb26/tv8394/workspace/zoology/data_kernel",
)

# ── Model factory ─────────────────────────────────────────────────────────────
conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={"l_max": input_seq_len, "kernel_size": 3, "implicit_long_conv": True},
)

model_factory_kwargs = {
    "state_mixer": dict(name="torch.nn.Identity", kwargs={}),
    "vocab_size": VOCAB_SIZE,
}

models = []

# ── Baseline: original DeltaNet ───────────────────────────────────────────────
for d_model in [64, 128, 256]:
    delta_net_mixer = dict(
        name="zoology.mixers.delta_net.DeltaNet",
        kwargs={
            "l_max": input_seq_len,
            "num_heads": 2,
            "use_beta": True,
            "use_gate": False,
            "use_short_conv": True,
            "conv_size": 4,
        }
    )
    mixer = ModuleConfig(
        name="zoology.mixers.hybrid.Hybrid",
        kwargs={"configs": [conv_mixer, delta_net_mixer]},
    )
    models.append(ModelConfig(
        block_type="TransformerBlock",
        d_model=d_model,
        n_layers=2,
        sequence_mixer=mixer,
        max_position_embeddings=0,
        name="delta_net_baseline",
        **model_factory_kwargs,
    ))

# ── Variant: DeltaNet + Taylor kernel ─────────────────────────────────────────
for d_model in [64, 128, 256]:
    for feature_dim in [8, 16]:
        kernel_mixer = dict(
            name="zoology.mixers.delta_net_kernel.DeltaNetKernel",
            kwargs={
                "num_heads": 2,
                "use_beta": True,
                "use_gate": False,
                "use_short_conv": True,
                "conv_size": 4,
                "feature_map": "taylor_exp",
                "feature_dim": feature_dim,
            }
        )
        mixer = ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": [conv_mixer, kernel_mixer]},
        )
        models.append(ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=2,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name=f"delta_net_taylor_fdim{feature_dim}",
            **model_factory_kwargs,
        ))

# ── Variant: DeltaNet + PosELU kernel ─────────────────────────────────────────
for d_model in [64, 128, 256]:
    kernel_mixer = dict(
        name="zoology.mixers.delta_net_kernel.DeltaNetKernel",
        kwargs={
            "num_heads": 2,
            "use_beta": True,
            "use_gate": False,
            "use_short_conv": True,
            "conv_size": 4,
            "feature_map": "pos_elu",
            "feature_dim": 32,  # pos_elu doesn't expand, so can use larger dim
        }
    )
    mixer = ModuleConfig(
        name="zoology.mixers.hybrid.Hybrid",
        kwargs={"configs": [conv_mixer, kernel_mixer]},
    )
    models.append(ModelConfig(
        block_type="TransformerBlock",
        d_model=d_model,
        n_layers=2,
        sequence_mixer=mixer,
        max_position_embeddings=0,
        name="delta_net_poselu",
        **model_factory_kwargs,
    ))

# ── Sweep over learning rates ──────────────────────────────────────────────────
configs = []
for model in models:
    for lr in np.logspace(-3, -1.5, 4):
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

print('==REMOVEME len configs', len(configs))