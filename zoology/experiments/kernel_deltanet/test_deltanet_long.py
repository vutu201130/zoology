"""
Test DeltaNet với seq_len dài (1024) để reproduce lỗi Triton.

Run:
    python -m zoology.launch zoology/experiments/kernel_deltanet/test_deltanet_long.py
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig
from zoology.data.multiquery_ar import MQARConfig

config = TrainConfig(
    data=DataConfig(
        train_configs=[
            MQARConfig(vocab_size=256, input_seq_len=256, num_examples=200, num_kv_pairs=16),
        ],
        test_configs=[
            MQARConfig(vocab_size=256, input_seq_len=512,  num_examples=50, num_kv_pairs=64),
            MQARConfig(vocab_size=256, input_seq_len=1024, num_examples=50, num_kv_pairs=128),
        ],
        batch_size=(16, 8),
    ),
    model=ModelConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        max_position_embeddings=0,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.delta_net.DeltaNet",
            kwargs={
                "l_max": 1024,
                "num_heads": 2,
                "use_beta": True,
                "use_gate": False,
                "use_short_conv": True,
                "conv_size": 4,
            }
        ),
        state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
    ),
    learning_rate=1e-3,
    max_epochs=2,
)

configs = [config]
