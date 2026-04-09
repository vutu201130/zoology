"""
Smoke test for run_baseline_gap.py — 1 config per model, tiny data, 2 epochs.

Run:
    python -m zoology.launch zoology/experiments/kernel_deltanet/test_baseline_gap.py
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB_SIZE = 8_192
input_seq_len = 1024

data = DataConfig(
    train_configs=[
        MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,  num_examples=200, num_kv_pairs=4),
        MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512, num_examples=200, num_kv_pairs=64),
    ],
    test_configs=[
        MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,   num_examples=50, num_kv_pairs=4),
        MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512,  num_examples=50, num_kv_pairs=128),
        MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024, num_examples=50, num_kv_pairs=256),
    ],
    batch_size=(16, 4),
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
    kwargs={"l_max": input_seq_len, "kernel_size": 3, "implicit_long_conv": True},
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
        max_epochs=2,
    )

configs = [
    make_config("attention", dict(
        name="zoology.mixers.attention.MHA",
        kwargs={"num_heads": 2},
    )),
    make_config("based", dict(
        name="zoology.mixers.based.Based",
        kwargs={"l_max": input_seq_len, "feature_dim": 8, "feature_name": "taylor_exp",
                "num_key_value_heads": 1, "num_heads": 1, "train_view": "quadratic"},
    )),
    make_config("delta_net", dict(
        name="zoology.mixers.delta_net.DeltaNet",
        kwargs={"l_max": input_seq_len, "num_heads": 2, "use_beta": True,
                "use_gate": False, "use_short_conv": True, "conv_size": 4},
    )),
    make_config("gated_delta_net", dict(
        name="zoology.mixers.gated_delta_net.GatedDeltaNet",
        kwargs={"num_heads": 2, "expand_v": 1, "use_gate": False,
                "use_short_conv": True, "conv_size": 4},
    )),
    make_config("gla", dict(
        name="zoology.mixers.gla.GatedLinearAttention",
        kwargs={"num_heads": 2, "use_short_conv": False},
    )),
]

print(f">> Total configs: {len(configs)}")
for c in configs:
    print(f"   {c.model.name}")
