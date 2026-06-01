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


@dataclass(frozen=True)
class MVTecTestImageRecord:
    path: Path
    category: str
    defect_type: str
    split: str = "test"
    label: int = 0
    mask_path: Optional[Path] = None


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


def list_mvtec_test_images(
    root,
    *,
    categories: Optional[Sequence[str]] = None,
    limit_per_category: Optional[int] = None,
) -> List[MVTecTestImageRecord]:
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"MVTec root does not exist: {root}")

    selected = tuple(categories or MVTEC_CLASSES)
    records: List[MVTecTestImageRecord] = []
    missing = []
    for category in selected:
        test_dir = root / category / "test"
        if not test_dir.is_dir():
            missing.append(str(test_dir))
            continue
        category_records: List[MVTecTestImageRecord] = []
        for defect_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
            defect_type = defect_dir.name
            label = 0 if defect_type == "good" else 1
            image_paths = sorted(path for path in defect_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
            for image_path in image_paths:
                mask_path = None
                if label:
                    mask_path = _find_mvtec_mask(root, category, defect_type, image_path)
                category_records.append(
                    MVTecTestImageRecord(
                        path=image_path,
                        category=category,
                        defect_type=defect_type,
                        label=label,
                        mask_path=mask_path,
                    )
                )
        if limit_per_category is not None:
            category_records = _limit_test_records(category_records, limit_per_category)
        records.extend(category_records)

    if missing and not records:
        raise FileNotFoundError("no MVTec test images found. Checked:\n" + "\n".join(missing))
    if not records:
        raise FileNotFoundError(f"no MVTec test images found under {root}")
    return records


def _find_mvtec_mask(root: Path, category: str, defect_type: str, image_path: Path) -> Path:
    mask_dir = root / category / "ground_truth" / defect_type
    candidates = [
        mask_dir / f"{image_path.stem}_mask.png",
        mask_dir / f"{image_path.stem}.png",
        mask_dir / f"{image_path.stem}_mask.bmp",
        mask_dir / f"{image_path.stem}.bmp",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing MVTec ground-truth mask for {image_path}")


def _limit_test_records(
    records: Sequence[MVTecTestImageRecord],
    limit_per_category: int,
) -> List[MVTecTestImageRecord]:
    limit = max(0, int(limit_per_category))
    if len(records) <= limit:
        return list(records)
    if limit >= 2:
        good = [record for record in records if record.label == 0]
        defects = [record for record in records if record.label == 1]
        if good and defects:
            selected = [defects[0], good[0]]
            selected_paths = {record.path for record in selected}
            for record in records:
                if len(selected) >= limit:
                    break
                if record.path not in selected_paths:
                    selected.append(record)
                    selected_paths.add(record.path)
            return selected
    return list(records[:limit])


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


class MVTecTestDataset(Dataset):
    def __init__(
        self,
        root,
        *,
        categories: Optional[Sequence[str]] = None,
        limit_per_category: Optional[int] = None,
    ) -> None:
        self.records = list_mvtec_test_images(
            root,
            categories=categories,
            limit_per_category=limit_per_category,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = Image.open(record.path).convert("RGB")
        if record.mask_path is None:
            mask = Image.new("L", image.size, color=0)
        else:
            mask = Image.open(record.mask_path).convert("L")
        return image, mask, {
            "path": str(record.path),
            "mask_path": None if record.mask_path is None else str(record.mask_path),
            "category": record.category,
            "defect_type": record.defect_type,
            "split": record.split,
            "label": record.label,
        }
