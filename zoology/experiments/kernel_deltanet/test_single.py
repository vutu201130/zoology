"""
Quick test: single DeltaNetKernel config to verify the code runs.

Run:
    python -m zoology.launch zoology/experiments/kernel_deltanet/test_single.py
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig
from zoology.data.multiquery_ar import MQARConfig

input_seq_len = 64

config = TrainConfig(
    data=DataConfig(
        train_configs=[MQARConfig(vocab_size=256, input_seq_len=input_seq_len, num_examples=1_000, num_kv_pairs=4)],
        test_configs=[MQARConfig(vocab_size=256, input_seq_len=input_seq_len, num_examples=200, num_kv_pairs=4)],
        batch_size=(32, 32),
    ),
    model=ModelConfig(
        vocab_size=256,
        max_position_embeddings=0,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.delta_net_kernel.DeltaNetKernel",
            kwargs={
                "num_heads": 2,
                "use_beta": True,
                "use_gate": False,
                "use_short_conv": True,
                "conv_size": 4,
                "feature_map": "taylor_exp",
                "feature_dim": 8,
            }
        ),
        state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
        d_model=64,
        n_layers=2,
    ),
    learning_rate=1e-3,
    max_epochs=3,
)

configs = [config]
