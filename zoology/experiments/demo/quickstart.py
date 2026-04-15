"""
Demo nhỏ để hiểu flow của zoology:
  1. Tạo MQAR data
  2. Tạo model với MHA (softmax attention)
  3. Train và đánh giá

Data rất nhỏ để chạy nhanh trên CPU/GPU bất kỳ.

Run:
    python -m zoology.launch zoology/zoology/experiments/demo/quickstart.py
    # hoặc train một config:
    python -m zoology.train --config path/to/config.yaml
"""

from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

# ── 1. DATA CONFIG ─────────────────────────────────────────────────────────────
# MQAR: Multi-Query Associative Recall
# - vocab_size: số token phân biệt
# - input_seq_len: độ dài sequence
# - num_kv_pairs: số cặp (key, value) cần nhớ — càng nhiều càng khó
# - num_examples: số training examples

VOCAB_SIZE = 256
SEQ_LEN = 64
KV_PAIRS = 4   # dễ — chỉ cần nhớ 4 cặp trong sequence 64 token

data = DataConfig(
    train_configs=[
        MQARConfig(
            vocab_size=VOCAB_SIZE,
            input_seq_len=SEQ_LEN,
            num_kv_pairs=KV_PAIRS,
            num_examples=2_000,    # nhỏ để chạy nhanh
        )
    ],
    test_configs=[
        MQARConfig(
            vocab_size=VOCAB_SIZE,
            input_seq_len=SEQ_LEN,
            num_kv_pairs=KV_PAIRS,
            num_examples=500,
        )
    ],
    batch_size=32,
    # cache_dir=None  → data được generate mỗi lần (không cache)
)

# ── 2. MODEL CONFIG ────────────────────────────────────────────────────────────
# ModelConfig định nghĩa kiến trúc:
#   - sequence_mixer: module xử lý sequence (attention, linear attention, SSM, ...)
#   - state_mixer: MLP sau mỗi block (mặc định hidden_mult=4)
#   - n_layers: số lớp transformer block
#   - d_model: hidden dimension

def make_model(name: str, mixer: dict) -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=SEQ_LEN,
        d_model=128,
        n_layers=2,
        sequence_mixer=ModuleConfig(name=mixer["name"], kwargs=mixer.get("kwargs", {})),
        name=name,
    )

# ── 3. ĐỊNH NGHĨA CÁC MODEL CẦN SO SÁNH ───────────────────────────────────────

MIXERS = {
    # Softmax attention — baseline mạnh nhất
    "softmax": {
        "name": "zoology.mixers.attention.MHA",
        "kwargs": {"num_heads": 2, "dropout": 0.0},
    },
    # Based với TaylorExp feature map — linear attention
    "based": {
        "name": "zoology.mixers.based.Based",
        "kwargs": {
            "l_max": SEQ_LEN,
            "feature_dim": 8,
            "feature_map": "taylor_exp",
            "num_heads": 1,
        },
    },
}

# ── 4. SWEEP LEARNING RATE ─────────────────────────────────────────────────────

configs = []

for name, mixer in MIXERS.items():
    for lr in [1e-3]:
        config = TrainConfig(
            data=data,
            model=make_model(name, mixer),
            learning_rate=lr,
            max_epochs=20,                      # nhỏ để chạy nhanh
            early_stopping_metric="valid/accuracy",
            early_stopping_threshold=0.99,
            run_id=f"{name}_lr{lr:.0e}",
            logger=LoggerConfig(
                project_name="zoology-demo",    # wandb project (chỉ log nếu wandb được setup)
            ),
        )
        configs.append(config)

print(f">> Tổng số configs: {len(configs)}")
for c in configs:
    print(f"   - {c.run_id}  (lr={c.learning_rate})")
