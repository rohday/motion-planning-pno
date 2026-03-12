from dataclasses import dataclass


@dataclass
class FNOConfig:
    modes1: int = 12
    modes2: int = 12
    width: int = 32
    num_layers: int = 4
    padding: int = 9
