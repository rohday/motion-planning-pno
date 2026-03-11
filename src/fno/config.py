from dataclasses import dataclass


@dataclass
class FNOConfig:
    """Configuration for the FNO2dMultiGoal model."""
    modes1: int = 12
    modes2: int = 12
    width: int = 32
    num_layers: int = 4
    padding: int = 9
