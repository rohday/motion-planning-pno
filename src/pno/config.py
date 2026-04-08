from dataclasses import dataclass


@dataclass
class PNOConfig:
    modes1: int = 12
    modes2: int = 12
    width: int = 48
    depth: int = 4
    padding: int = 9
    beta: float = 5.0
    deepnorm_hidden: int = 64
