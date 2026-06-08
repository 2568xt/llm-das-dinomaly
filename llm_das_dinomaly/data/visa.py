from __future__ import annotations

from typing import Optional, Sequence

from llm_das_dinomaly.data.mvtec import MVTecGoodDataset, MVTecTestDataset
from llm_das_dinomaly.data.mvtec import list_mvtec_test_images, list_mvtec_train_good


VISA_CLASSES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)


def list_visa_train_good(
    root,
    *,
    categories: Optional[Sequence[str]] = None,
    limit_per_category: Optional[int] = None,
):
    return list_mvtec_train_good(
        root,
        categories=categories or VISA_CLASSES,
        limit_per_category=limit_per_category,
    )


def list_visa_test_images(
    root,
    *,
    categories: Optional[Sequence[str]] = None,
    limit_per_category: Optional[int] = None,
):
    return list_mvtec_test_images(
        root,
        categories=categories or VISA_CLASSES,
        limit_per_category=limit_per_category,
    )


class ViSAGoodDataset(MVTecGoodDataset):
    def __init__(
        self,
        root,
        *,
        categories: Optional[Sequence[str]] = None,
        limit_per_category: Optional[int] = None,
    ) -> None:
        self.records = list_visa_train_good(
            root,
            categories=categories,
            limit_per_category=limit_per_category,
        )


class ViSATestDataset(MVTecTestDataset):
    def __init__(
        self,
        root,
        *,
        categories: Optional[Sequence[str]] = None,
        limit_per_category: Optional[int] = None,
    ) -> None:
        self.records = list_visa_test_images(
            root,
            categories=categories,
            limit_per_category=limit_per_category,
        )
