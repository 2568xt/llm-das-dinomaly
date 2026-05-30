from __future__ import annotations

import contextlib
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn

from llm_das_dinomaly.wrappers import DinomalyConfig, DinomalyWrapper


class DinomalyLoadError(RuntimeError):
    pass


def build_dinomaly_wrapper(
    *,
    dinomaly_root: Union[str, Path],
    checkpoint_path: Union[str, Path],
    device: str = "cuda",
    backbone: str = "dinov2reg_vit_base_14",
    strict: bool = False,
) -> Tuple[DinomalyWrapper, Dict[str, Any]]:
    """Build official Dinomaly and wrap it with `DinomalyWrapper`.

    This intentionally mirrors the official `dinomaly_mvtec_uni.py` model
    construction while keeping all imports isolated to the configured submodule.
    """

    root = Path(dinomaly_root).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not root.exists():
        raise DinomalyLoadError(f"DINOMALY_ROOT does not exist: {root}")
    if not (root / "models" / "uad.py").exists():
        raise DinomalyLoadError(f"DINOMALY_ROOT does not look like the official repo: {root}")
    if not checkpoint.is_file():
        raise DinomalyLoadError(f"CHECKPOINT_PATH does not exist or is not a file: {checkpoint}")

    with _dinomaly_import_context(root):
        try:
            from dinov1.utils import trunc_normal_
            from models import vit_encoder
            from models.uad import ViTill
            from models.vision_transformer import LinearAttention2
            from models.vision_transformer import Block as VitBlock
            from models.vision_transformer import bMlp
        except Exception as exc:  # pragma: no cover - depends on server deps.
            raise DinomalyLoadError(
                "failed to import official Dinomaly modules. Install its requirements in the server env."
            ) from exc

        target_layers, fuse_encoder, fuse_decoder, embed_dim, num_heads = _architecture(backbone)
        encoder = vit_encoder.load(backbone)
        bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])
        decoder = nn.ModuleList(
            [
                VitBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-8),
                    attn=LinearAttention2,
                )
                for _ in range(8)
            ]
        )
        model = ViTill(
            encoder=encoder,
            bottleneck=bottleneck,
            decoder=decoder,
            target_layers=target_layers,
            mask_neighbor_size=0,
            fuse_layer_encoder=fuse_encoder,
            fuse_layer_decoder=fuse_decoder,
        )

        _init_trainable_layers(model, trunc_normal_)

    state_dict = _extract_state_dict(torch.load(checkpoint, map_location="cpu"))
    load_result = model.load_state_dict(state_dict, strict=strict)
    model = model.to(device).eval()

    cfg = DinomalyConfig(
        backbone=backbone,
        target_layers=tuple(target_layers),
        fuse_layer_encoder=tuple(tuple(group) for group in fuse_encoder),
        fuse_layer_decoder=tuple(tuple(group) for group in fuse_decoder),
        device=device,
    )
    wrapper = DinomalyWrapper(model, cfg).to(device).eval()
    metadata = {
        "dinomaly_root": str(root),
        "checkpoint_path": str(checkpoint),
        "backbone": backbone,
        "strict": strict,
        "missing_keys": list(getattr(load_result, "missing_keys", [])),
        "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
        "wrapper": wrapper.metadata(),
    }
    return wrapper, metadata


def _architecture(backbone: str):
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    if "small" in backbone:
        return target_layers, fuse_encoder, fuse_decoder, 384, 6
    if "base" in backbone:
        return target_layers, fuse_encoder, fuse_decoder, 768, 12
    if "large" in backbone:
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        return target_layers, fuse_encoder, fuse_decoder, 1024, 16
    raise DinomalyLoadError(f"unsupported Dinomaly backbone: {backbone}")


def _init_trainable_layers(model: nn.Module, trunc_normal_) -> None:
    trainable = nn.ModuleList([model.bottleneck, model.decoder])
    for module in trainable.modules():
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "net", "module"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise DinomalyLoadError("checkpoint must be a state_dict or contain state_dict/model/net/module")
    out = {}
    for key, value in checkpoint.items():
        name = str(key)
        if name.startswith("module."):
            name = name[len("module.") :]
        out[name] = value
    return out


@contextlib.contextmanager
def _dinomaly_import_context(root: Path):
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    sys.path.insert(0, str(root))
    try:
        import os

        os.chdir(root)
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
