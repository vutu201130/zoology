"""
Compare: DeltaNet (L2) vs DeltaNet (ReBased) vs GatedDeltaNet (L2) vs GatedDeltaNet (ReBased).

Data: cùng cache_dir với test_gla_vs_softmax.py — không generate lại data.
  seq_len=512, num_kv_pairs=64, 100k train, 3k test, batch=128, 64 epochs.

d_model=128, 2 layers, lr sweep 4 values.

Run:
    python -m zoology.launch zoology/experiments/rebased/run_deltanet_vs_rebased.py -p
"""
import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB_SIZE = 8_192
INPUT_SEQ_LEN = 512

sweep_id = uuid.uuid4().hex[:6]
sweep_name = "deltanet_vs_rebased_" + sweep_id

# Same data config as test_gla_vs_softmax.py — reuses cached data
data = DataConfig(
    train_configs=[
        MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=INPUT_SEQ_LEN,
                   num_examples=100_000, num_kv_pairs=64),
    ],
    test_configs=[
        MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=INPUT_SEQ_LEN,
                   num_examples=3_000, num_kv_pairs=64),
    ],
    batch_size=(128, 32),
    cache_dir="/scratch/mb26/tv8394/workspace/zoology/data_gla_vs_softmax",
)

conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={"l_max": INPUT_SEQ_LEN, "kernel_size": 3, "implicit_long_conv": True},
)

base_model_kwargs = dict(
    vocab_size=VOCAB_SIZE,
    d_model=128,
    n_layers=2,
    block_type="TransformerBlock",
    max_position_embeddings=0,
    state_mixer=dict(name="torch.nn.Identity", kwargs={}),
)


def make_config(name, mixer_dict, lr):
    return TrainConfig(
        data=data,
        model=ModelConfig(
            name=name,
            sequence_mixer=ModuleConfig(
                name="zoology.mixers.hybrid.Hybrid",
                kwargs={"configs": [conv_mixer, mixer_dict]},
            ),
            **base_model_kwargs,
        ),
        learning_rate=lr,
        max_epochs=64,
        logger=LoggerConfig(
            project_name="zoology",
            entity="vutu201130-matrixone",
            tags=["deltanet-vs-rebased", "rebased-kernel", name],
        ),
        slice_keys=["num_kv_pairs"],
        sweep_id=sweep_name,
        run_id=f"{name}-d128-lr{lr:.1e}",
    )


MIXERS = {
    "delta_net_l2": dict(
        name="zoology.mixers.delta_net.DeltaNet",
        kwargs={"l_max": INPUT_SEQ_LEN, "num_heads": 2, "use_beta": True,
                "use_gate": False, "use_short_conv": True, "conv_size": 4},
    ),
    "delta_net_rb": dict(
        name="zoology.mixers.delta_net_rebased.DeltaNetReBased",
        kwargs={"num_heads": 2, "feature_dim": 11,
                "use_beta": True, "use_gate": False,
                "use_short_conv": True, "conv_size": 4},
    ),
    "gated_delta_l2": dict(
        name="zoology.mixers.gated_delta_net.GatedDeltaNet",
        kwargs={"num_heads": 2, "expand_v": 1, "use_gate": False,
                "use_short_conv": True, "conv_size": 4},
    ),
    "gated_delta_rb": dict(
        name="zoology.mixers.gated_delta_net_rebased.GatedDeltaNetReBased",
        kwargs={"num_heads": 2, "feature_dim": 11, "expand_v": 1,
                "use_gate": False, "use_short_conv": True, "conv_size": 4},
    ),
}

configs = []
for name, mixer_dict in MIXERS.items():
    for lr in np.logspace(-3, -1.5, 4):
        configs.append(make_config(name, mixer_dict, lr))

print(f">> Total configs: {len(configs)}")
for c in configs:
    print(f"   {c.run_id}")
