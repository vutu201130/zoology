"""
Smoke test: RetNet vs RetNetReBased forward/backward pass.

seq_len=512, num_kv_pairs=64, d_model=128, 2 epochs.

Run:
    python -m zoology.launch zoology/experiments/rebased/test_retnet_vs_rebased.py
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
        max_epochs=2,
    )


configs = [
    make_config("retnet", dict(
        name="zoology.mixers.retnet.RetNet",
        kwargs={"num_heads": 2},
    )),
    make_config("retnet_rb", dict(
        name="zoology.mixers.retnet_rebased.RetNetReBased",
        kwargs={"num_heads": 2, "feature_dim": 11},
    )),
]

print(f">> Total configs: {len(configs)}")
for c in configs:
    print(f"   {c.model.name}")
