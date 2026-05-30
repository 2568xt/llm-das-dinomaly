from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import torch


def save_tensor_cache(path: Union[str, Path], tensors: Dict[str, torch.Tensor], metadata: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tensors": {key: value.detach().cpu() for key, value in tensors.items()},
        "metadata": metadata,
    }
    torch.save(payload, path)


def load_tensor_cache(path: Union[str, Path]) -> Dict[str, Any]:
    return torch.load(Path(path), map_location="cpu")
