from __future__ import annotations

from typing import Optional, Sequence

from llm_das_dinomaly.data.mvtec import MVTecGoodDataset, MVTecTestDataset
from llm_das_dinomaly.data.mvtec import list_mvtec_test_images, list_mvtec_train_good


MPDD_CLASSES = (
    "bracket_black",
    "bracket_brown",
    "bracket_white",
    "connector",
    "metal_plate",
    "tubes",
)


def list_mpdd_train_good(
    root,
    *,
    categories: Optional[Sequence[str]] = None,
    limit_per_category: Optional[int] = None,
):
    return list_mvtec_train_good(
        root,
        categories=categories or MPDD_CLASSES,
        limit_per_category=limit_per_category,
    )


def list_mpdd_test_images(
    root,
    *,
    categories: Optional[Sequence[str]] = None,
    limit_per_category: Optional[int] = None,
):
    return list_mvtec_test_images(
        root,
        categories=categories or MPDD_CLASSES,
        limit_per_category=limit_per_category,
    )


class MPDDGoodDataset(MVTecGoodDataset):
    def __init__(
        self,
        root,
        *,
        categories: Optional[Sequence[str]] = None,
        limit_per_category: Optional[int] = None,
    ) -> None:
        self.records = list_mpdd_train_good(
            root,
            categories=categories,
            limit_per_category=limit_per_category,
        )


class MPDDTestDataset(MVTecTestDataset):
    def __init__(
        self,
        root,
        *,
        categories: Optional[Sequence[str]] = None,
        limit_per_category: Optional[int] = None,
    ) -> None:
        self.records = list_mpdd_test_images(
            root,
            categories=categories,
            limit_per_category=limit_per_category,
        )
