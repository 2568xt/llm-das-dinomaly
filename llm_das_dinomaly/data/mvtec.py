from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MVTEC_CLASSES = (
    "carpet",
    "grid",
    "leather",
    "tile",
    "wood",
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
)


@dataclass(frozen=True)
class MVTecImageRecord:
    path: Path
    category: str
    split: str = "train"
    label: int = 0


def list_mvtec_train_good(
    root,
    *,
    categories: Optional[Sequence[str]] = None,
    limit_per_category: Optional[int] = None,
) -> List[MVTecImageRecord]:
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"MVTec root does not exist: {root}")

    selected = tuple(categories or MVTEC_CLASSES)
    records: List[MVTecImageRecord] = []
    missing = []
    for category in selected:
        good_dir = root / category / "train" / "good"
        if not good_dir.is_dir():
            missing.append(str(good_dir))
            continue
        paths = sorted(path for path in good_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        if limit_per_category is not None:
            paths = paths[:limit_per_category]
        records.extend(MVTecImageRecord(path=path, category=category) for path in paths)

    if missing and not records:
        raise FileNotFoundError("no MVTec train/good images found. Checked:\n" + "\n".join(missing))
    if not records:
        raise FileNotFoundError(f"no MVTec train/good images found under {root}")
    return records


class MVTecGoodDataset(Dataset):
    def __init__(
        self,
        root,
        *,
        categories: Optional[Sequence[str]] = None,
        limit_per_category: Optional[int] = None,
    ) -> None:
        self.records = list_mvtec_train_good(
            root,
            categories=categories,
            limit_per_category=limit_per_category,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = Image.open(record.path).convert("RGB")
        return image, {
            "path": str(record.path),
            "category": record.category,
            "label": record.label,
        }
