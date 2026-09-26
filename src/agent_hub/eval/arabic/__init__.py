"""Arabic text handling for the eval harness (Domain 2)."""

from agent_hub.eval.arabic.normalize import (
    AGGRESSIVE,
    CONSERVATIVE,
    OFF,
    NormalizeConfig,
    normalize,
    per_transform_variants,
)

__all__ = [
    "AGGRESSIVE",
    "CONSERVATIVE",
    "OFF",
    "NormalizeConfig",
    "normalize",
    "per_transform_variants",
]
