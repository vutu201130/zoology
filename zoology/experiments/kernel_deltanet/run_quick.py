"""
Quick sweep: 3 models x 1 d_model x 2 LRs = 6 configs (vs 48 in run.py).
Use this to verify all variants run correctly before launching full sweep.

Run:
    python -m zoology.launch zoology/experiments/kernel_deltanet/run_quick.py -p
"""
import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

sweep_id = uuid.uuid4().hex[:6]
sweep_name = "kernel_deltanet_quick_" + sweep_id

VOCAB_SIZE = 8_192

# ── Data (smaller than full run) ──────────────────────────────────────────────
train_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,  num_examples=10_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=10_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=10_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=10_000, num_kv_pairs=32),
]
test_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,  num_examples=500, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=500, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=500, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=500, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512, num_examples=500, num_kv_pairs=64),
]

input_seq_len = max(c.input_seq_len for c in train_configs + test_configs)
data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    batch_size=(64, 64),
    cache_dir="/scratch/mb26/tv8394/workspace/zoology/data_kernel_quick",
)

conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={"l_max": input_seq_len, "kernel_size": 3, "implicit_long_conv": True},
)

model_factory_kwargs = {
    "state_mixer": dict(name="torch.nn.Identity", kwargs={}),
    "vocab_size": VOCAB_SIZE,
    "block_type": "TransformerBlock",
    "d_model": 128,
    "n_layers": 2,
    "max_position_embeddings": 0,
}

models = [
    # Baseline
    ModelConfig(
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": [conv_mixer, dict(
                name="zoology.mixers.delta_net.DeltaNet",
                kwargs={"l_max": input_seq_len, "num_heads": 2, "use_beta": True,
                        "use_gate": False, "use_short_conv": True, "conv_size": 4}
            )]},
        ),
        name="delta_net_baseline",
        **model_factory_kwargs,
    ),
    # Taylor fdim=8
    ModelConfig(
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": [conv_mixer, dict(
                name="zoology.mixers.delta_net_kernel.DeltaNetKernel",
                kwargs={"num_heads": 2, "use_beta": True, "use_gate": False,
                        "use_short_conv": True, "conv_size": 4,
                        "feature_map": "taylor_exp", "feature_dim": 8}
            )]},
        ),
        name="delta_net_taylor_fdim8",
        **model_factory_kwargs,
    ),
    # PosELU
    ModelConfig(
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": [conv_mixer, dict(
                name="zoology.mixers.delta_net_kernel.DeltaNetKernel",
                kwargs={"num_heads": 2, "use_beta": True, "use_gate": False,
                        "use_short_conv": True, "conv_size": 4,
                        "feature_map": "pos_elu", "feature_dim": 32}
            )]},
        ),
        name="delta_net_poselu",
        **model_factory_kwargs,
    ),
]

configs = []
for model in models:
    for lr in [1e-3, 3e-3]:
        run_id = f"{model.name}-d{model.d_model}-lr{lr:.1e}"
        configs.append(TrainConfig(
            model=model,
            data=data,
            learning_rate=lr,
            max_epochs=16,
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
