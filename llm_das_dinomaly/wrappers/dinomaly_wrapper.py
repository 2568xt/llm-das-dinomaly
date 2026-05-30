from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

Tensor = torch.Tensor


@dataclass(frozen=True)
class DinomalyConfig:
    """Stable public configuration for a Dinomaly detector wrapper."""

    backbone: str = "dinov2reg_vit_base_14"
    image_size: int = 448
    crop_size: int = 392
    patch_size: int = 14
    resize_mask: Optional[int] = 256
    target_layers: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9)
    fuse_layer_encoder: Tuple[Tuple[int, ...], ...] = ((0, 1, 2, 3), (4, 5, 6, 7))
    fuse_layer_decoder: Tuple[Tuple[int, ...], ...] = ((0, 1, 2, 3), (4, 5, 6, 7))
    bottleneck_dropout: float = 0.2
    topk_ratio: float = 0.01
    gaussian_kernel: int = 5
    gaussian_sigma: float = 4.0
    device: str = "cuda"

    @property
    def feature_size(self) -> int:
        if self.crop_size % self.patch_size != 0:
            raise ValueError("crop_size must be divisible by patch_size")
        return self.crop_size // self.patch_size

    @property
    def expected_channels(self) -> int:
        if "small" in self.backbone:
            return 384
        if "large" in self.backbone:
            return 1024
        return 768

    def to_metadata(self) -> Dict[str, Any]:
        return asdict(self)


class DinomalyWrapper(nn.Module):
    """Detector-style API around a Dinomaly-compatible model.

    The wrapped model should return either `(encoder_groups, decoder_groups)` or
    a dict containing `encoder_groups` and `decoder_groups`. Each group can be a
    feature map `[B, C, H, W]` or square patch tokens `[B, N, C]`.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Optional[DinomalyConfig] = None,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.model = model.eval()
        self.cfg = cfg or DinomalyConfig()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def preprocess(
        self,
        images: Union[Tensor, Sequence[Image.Image]],
        *,
        normalize: bool = True,
    ) -> Tensor:
        """Convert images to `[B, 3, crop_size, crop_size]`.

        Tensor inputs are expected in `[0, 1]` if floating point or `[0, 255]`
        if integer typed. PIL inputs are converted to RGB.
        """

        x = self._to_bchw(images).to(device=self.mean.device, dtype=self.mean.dtype)
        x = F.interpolate(
            x,
            size=(self.cfg.image_size, self.cfg.image_size),
            mode="bilinear",
            align_corners=False,
        )
        x = self._center_crop(x, self.cfg.crop_size)
        if normalize:
            x = (x - self.mean) / self.std
        return x

    def denormalize(self, x: Tensor) -> Tensor:
        return x * self.std.to(x.device, x.dtype) + self.mean.to(x.device, x.dtype)

    @torch.no_grad()
    def forward_features(self, x: Tensor) -> Dict[str, List[Tensor]]:
        x = x.to(device=self.mean.device, dtype=self.mean.dtype)
        out = self.model(x)
        encoder_groups, decoder_groups = self._unpack_model_output(out)
        return {
            "encoder_groups": [self._coerce_group_map(t) for t in encoder_groups],
            "decoder_groups": [self._coerce_group_map(t) for t in decoder_groups],
        }

    @torch.no_grad()
    def extract_features(
        self,
        x: Tensor,
        *,
        which: Literal["encoder", "decoder", "both"] = "encoder",
        flatten_tokens: bool = False,
    ) -> Union[List[Tensor], Dict[str, List[Tensor]]]:
        out = self.forward_features(x)
        if flatten_tokens:
            out = {
                key: [feat.flatten(2).transpose(1, 2).contiguous() for feat in feats]
                for key, feats in out.items()
            }
        if which == "encoder":
            return out["encoder_groups"]
        if which == "decoder":
            return out["decoder_groups"]
        return out

    @torch.no_grad()
    def predict_map(
        self,
        x: Tensor,
        *,
        resize_to: Optional[int] = None,
        smooth: bool = True,
    ) -> Tensor:
        out = self.forward_features(x)
        target_size = x.shape[-1] if resize_to is None else resize_to
        anomaly_map = self._cosine_anomaly_map(
            out["encoder_groups"],
            out["decoder_groups"],
            out_size=target_size,
        )
        if smooth:
            anomaly_map = self._gaussian_blur(
                anomaly_map,
                kernel_size=self.cfg.gaussian_kernel,
                sigma=self.cfg.gaussian_sigma,
            )
        return anomaly_map

    @torch.no_grad()
    def predict_score(
        self,
        x: Tensor,
        *,
        topk_ratio: Optional[float] = None,
        resize_to: Optional[int] = None,
    ) -> Tensor:
        anomaly_map = self.predict_map(x, resize_to=resize_to, smooth=True)
        flat = anomaly_map.flatten(1)
        ratio = self.cfg.topk_ratio if topk_ratio is None else topk_ratio
        k = max(1, min(flat.shape[1], int(flat.shape[1] * ratio)))
        return torch.topk(flat, k=k, dim=1).values.mean(dim=1)

    @torch.no_grad()
    def score_candidates(
        self,
        x_ref: Tensor,
        x_cands: Tensor,
        *,
        synth_masks: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Score candidates against their reference normal image."""

        score_ref = self.predict_score(x_ref)
        score_cand = self.predict_score(x_cands)
        anomaly_map = self.predict_map(x_cands)
        out: Dict[str, Tensor] = {
            "score_ref": score_ref,
            "score_cand": score_cand,
            "score_delta": score_cand - score_ref,
            "map": anomaly_map,
            "perturb_l1": (x_cands - x_ref).abs().flatten(1).mean(1),
        }
        if synth_masks is not None:
            mask = self._mask_to_b1hw(synth_masks, x_cands.shape[0], x_cands.device, x_cands.dtype)
            mask = F.interpolate(mask, size=anomaly_map.shape[-2:], mode="nearest")
            anomaly_norm = self._minmax_by_sample(anomaly_map)
            mask_area = mask.flatten(1).mean(1)
            out["mask_area"] = mask_area
            out["mask_overlap"] = (anomaly_norm * mask).flatten(1).sum(1) / (
                mask.flatten(1).sum(1) + 1e-6
            )
        return out

    def metadata(self) -> Dict[str, Any]:
        return {
            "wrapper": self.__class__.__name__,
            "config": self.cfg.to_metadata(),
            "mean": self.mean.flatten().detach().cpu().tolist(),
            "std": self.std.flatten().detach().cpu().tolist(),
        }

    @staticmethod
    def _center_crop(x: Tensor, crop_size: int) -> Tensor:
        if x.shape[-2] < crop_size or x.shape[-1] < crop_size:
            raise ValueError("crop_size cannot exceed image dimensions")
        top = (x.shape[-2] - crop_size) // 2
        left = (x.shape[-1] - crop_size) // 2
        return x[..., top : top + crop_size, left : left + crop_size]

    @staticmethod
    def _minmax_by_sample(x: Tensor, eps: float = 1e-6) -> Tensor:
        flat = x.flatten(1)
        lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
        hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
        return (x - lo) / (hi - lo + eps)

    @staticmethod
    def _mask_to_b1hw(mask: Tensor, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        mask = mask.to(device=device, dtype=dtype)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)
        elif mask.ndim != 4:
            raise ValueError("mask must have shape [H,W], [B,H,W], or [B,1,H,W]")
        if mask.shape[1] != 1:
            mask = mask[:, :1]
        if mask.shape[0] == 1 and batch > 1:
            mask = mask.expand(batch, -1, -1, -1)
        if mask.shape[0] != batch:
            raise ValueError("mask batch size must be 1 or match candidates")
        return mask

    def _to_bchw(self, images: Union[Tensor, Sequence[Image.Image]]) -> Tensor:
        if isinstance(images, torch.Tensor):
            x = images
            if x.ndim == 3:
                x = x.unsqueeze(0)
            if x.ndim != 4:
                raise ValueError("tensor images must have shape [B,C,H,W] or [C,H,W]")
            if x.shape[1] == 1:
                x = x.expand(-1, 3, -1, -1)
            if x.shape[1] != 3:
                raise ValueError("tensor images must have 1 or 3 channels")
            x = x.float()
            if not torch.is_floating_point(images) or x.max().item() > 2.0:
                x = x / 255.0
            return x.clamp(0.0, 1.0)

        tensors: List[Tensor] = []
        for image in images:
            if not isinstance(image, Image.Image):
                raise TypeError("image sequences must contain PIL.Image.Image objects")
            arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            tensor = F.interpolate(
                tensor,
                size=(self.cfg.image_size, self.cfg.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            tensors.append(tensor)
        if not tensors:
            raise ValueError("at least one image is required")
        return torch.stack(tensors, dim=0)

    @staticmethod
    def _unpack_model_output(out: Any) -> Tuple[Sequence[Tensor], Sequence[Tensor]]:
        if isinstance(out, dict):
            encoder = out.get("encoder_groups", out.get("encoder"))
            decoder = out.get("decoder_groups", out.get("decoder"))
        elif isinstance(out, (tuple, list)) and len(out) == 2:
            encoder, decoder = out
        else:
            raise TypeError("model must return dict or (encoder_groups, decoder_groups)")

        if not isinstance(encoder, (list, tuple)) or not isinstance(decoder, (list, tuple)):
            raise TypeError("encoder and decoder outputs must be lists or tuples")
        if len(encoder) != len(decoder):
            raise ValueError("encoder and decoder group counts must match")
        return encoder, decoder

    @staticmethod
    def _coerce_group_map(group: Tensor) -> Tensor:
        if group.ndim == 4:
            return group
        if group.ndim != 3:
            raise ValueError("feature groups must be [B,C,H,W], [B,N,C], or [B,C,N]")

        b, a, c = group.shape
        side_a = int(sqrt(a))
        if side_a * side_a == a:
            return group.transpose(1, 2).reshape(b, c, side_a, side_a).contiguous()

        side_c = int(sqrt(c))
        if side_c * side_c == c:
            return group.reshape(b, a, side_c, side_c).contiguous()

        raise ValueError("token feature groups must contain a square patch dimension")

    def _cosine_anomaly_map(
        self,
        encoder_groups: Sequence[Tensor],
        decoder_groups: Sequence[Tensor],
        *,
        out_size: int,
    ) -> Tensor:
        maps = []
        for enc, dec in zip(encoder_groups, decoder_groups):
            if enc.shape[1] != dec.shape[1]:
                raise ValueError("encoder and decoder channel dimensions must match")
            if enc.shape[-2:] != dec.shape[-2:]:
                dec = F.interpolate(dec, size=enc.shape[-2:], mode="bilinear", align_corners=False)
            group_map = 1.0 - F.cosine_similarity(enc, dec, dim=1, eps=1e-6)
            group_map = group_map.unsqueeze(1)
            group_map = F.interpolate(
                group_map,
                size=(out_size, out_size),
                mode="bilinear",
                align_corners=False,
            )
            maps.append(group_map)
        return torch.stack(maps, dim=0).mean(dim=0)

    @staticmethod
    def _gaussian_blur(x: Tensor, *, kernel_size: int, sigma: float) -> Tensor:
        if kernel_size <= 1:
            return x
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = max(float(sigma), 1e-6)
        coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).expand(x.shape[1], 1, -1, -1)
        pad = kernel_size // 2
        mode = "reflect" if x.shape[-1] > pad and x.shape[-2] > pad else "replicate"
        x_pad = F.pad(x, (pad, pad, pad, pad), mode=mode)
        return F.conv2d(x_pad, kernel, groups=x.shape[1])
