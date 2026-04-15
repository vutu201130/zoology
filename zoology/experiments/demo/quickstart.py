"""
Demo nhỏ để hiểu flow của zoology.
Config lấy từ paper ICLR24 (figure2/configs.py):
  - Based = Hybrid(BaseConv + Based/TaylorExp)
  - 1 learning rate duy nhất
  - data nhỏ để chạy nhanh

Run:
    python -m zoology.launch zoology/zoology/experiments/demo/quickstart.py
"""

from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB_SIZE = 8_192
SEQ_LEN = 64
KV_PAIRS = 4

# ── DATA ───────────────────────────────────────────────────────────────────────
data = DataConfig(
    train_configs=[
        MQARConfig(
            num_examples=10_000,
            vocab_size=VOCAB_SIZE,
            input_seq_len=SEQ_LEN,
            num_kv_pairs=KV_PAIRS,
            train_power_a=0.01,
            test_power_a=0.01,
            random_non_queries=False,
        )
    ],
    test_configs=[
        MQARConfig(
            num_examples=1_000,
            vocab_size=VOCAB_SIZE,
            input_seq_len=SEQ_LEN,
            num_kv_pairs=KV_PAIRS,
            train_power_a=0.01,
            test_power_a=0.01,
            random_non_queries=False,
        )
    ],
    batch_size=512,
)

# ── MIXER: Hybrid(BaseConv + Based) — lấy thẳng từ paper ────────────────────
based_mixer = dict(
    name="zoology.mixers.hybrid.Hybrid",
    kwargs={
        "configs": [
            dict(
                name="zoology.mixers.base_conv.BaseConv",
                kwargs={
                    "l_max": SEQ_LEN,
                    "kernel_size": 3,
                    "implicit_long_conv": True,
                }
            ),
            dict(
                name="zoology.mixers.based.Based",
                kwargs={
                    "l_max": SEQ_LEN,
                    "feature_dim": 8,
                    "num_key_value_heads": 1,
                    "num_heads": 1,
                    "feature_name": "taylor_exp",
                    "train_view": "quadratic",
                }
            ),
        ]
    },
)

# ── MODEL ──────────────────────────────────────────────────────────────────────
model = ModelConfig(
    d_model=128,
    n_layers=2,
    block_type="TransformerBlock",
    max_position_embeddings=0,   # 0 = không dùng positional embedding (Based không cần)
    vocab_size=VOCAB_SIZE,
    sequence_mixer=based_mixer,
    state_mixer=dict(name="torch.nn.Identity", kwargs={}),  # bỏ MLP như paper
)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
configs = [
    TrainConfig(
        model=model,
        data=data,
        learning_rate=1e-3,
        max_epochs=64,
        run_id="based-demo-seqlen64-dmodel128-lr1e-3",
        logger=LoggerConfig(project_name="zoology-demo"),
    )
]

print(f">> Configs: {len(configs)}")
for c in configs:
    print(f"   {c.run_id}  lr={c.learning_rate}")
