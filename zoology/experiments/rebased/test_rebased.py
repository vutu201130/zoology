"""
Smoke test: verify ReBased mixer variants forward/backward pass without errors.

Data: seq_len=512, num_kv_pairs=64, vocab=8192 — hardest Zoology setting.
Model: d_model=128, 2 layers, batch=4, 2 epochs.

Run:
    python -m zoology.launch zoology/experiments/rebased/test_rebased.py
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB_SIZE = 8_192
INPUT_SEQ_LEN = 512

data = DataConfig(
    train_configs=[
        MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=INPUT_SEQ_LEN, num_examples=256, num_kv_pairs=64),
    ],
    test_configs=[
        MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=INPUT_SEQ_LEN, num_examples=64, num_kv_pairs=64),
    ],
    batch_size=(4, 4),
)

base_kwargs = dict(
    vocab_size=VOCAB_SIZE,
    d_model=128,
    n_layers=2,
    block_type="TransformerBlock",
    max_position_embeddings=0,
    state_mixer=dict(name="torch.nn.Identity", kwargs={}),
)

conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={"l_max": INPUT_SEQ_LEN, "kernel_size": 3, "implicit_long_conv": True},
)


def make_config(name, mixer_dict):
    return TrainConfig(
        data=data,
        model=ModelConfig(
            name=name,
            sequence_mixer=ModuleConfig(
                name="zoology.mixers.hybrid.Hybrid",
                kwargs={"configs": [conv_mixer, mixer_dict]},
            ),
            **base_kwargs,
        ),
        learning_rate=1e-3,
        max_epochs=1,
    )


configs = [
    make_config("delta_net_rb", dict(
        name="zoology.mixers.delta_net_rebased.DeltaNetReBased",
        kwargs={"num_heads": 2, "feature_dim": 11,
                "use_beta": True, "use_gate": False,
                "use_short_conv": True, "conv_size": 4},
    )),
    make_config("gated_delta_rb", dict(
        name="zoology.mixers.gated_delta_net_rebased.GatedDeltaNetReBased",
        kwargs={"num_heads": 2, "feature_dim": 11, "expand_v": 1,
                "use_gate": False, "use_short_conv": True, "conv_size": 4},
    )),
    make_config("gla_rb", dict(
        name="zoology.mixers.gla_rebased.GLAReBased",
        kwargs={"num_heads": 2, "feature_dim": 11},
    )),
]

print(f">> Total configs: {len(configs)}")
for c in configs:
    print(f"   {c.model.name}")
