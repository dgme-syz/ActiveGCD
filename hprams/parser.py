from typing import Any

from .train_config import TrainConfig
from .data_config import DatasetConfig


def split_params(params: dict[str, Any]) -> tuple[TrainConfig, DatasetConfig]:
    """ split the params into TrainConfig and DatasetConfig """
    
    train_params, data_params = {}, {}
    for key, value in params.items():
        if key in TrainConfig.__annotations__:
            train_params[key] = value
        elif key in DatasetConfig.__annotations__:
            data_params[key] = value
        else:
            raise ValueError(f"Unknown parameter: {key}")
    
    return TrainConfig(**train_params), DatasetConfig(**data_params)