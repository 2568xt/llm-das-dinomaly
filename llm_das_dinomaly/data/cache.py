from __future__ import annotations

import os
import tempfile
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
    save_torch_payload(path, payload)


def load_tensor_cache(path: Union[str, Path]) -> Dict[str, Any]:
    return torch.load(Path(path), map_location="cpu")


def save_torch_payload(path: Union[str, Path], payload: Any) -> None:
    """Atomically save a torch payload next to the final destination."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
        torch.save(payload, tmp_name)
        os.replace(tmp_name, path)
    finally:
        if tmp_name is not None:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass
