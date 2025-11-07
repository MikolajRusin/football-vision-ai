from .load_dataset import LoadDataset
from torch.utils.data import DataLoader
from pathlib import Path
import albumentations as A
from typing import Any
import torch

def collate_fn(batch) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    images = list(images)
    targets = [t for t in targets]
    return images, targets

def load_dataloader(root_dir: Path | str, coco_path: Path | str, set_ratio: float | int | None = None, custom_categories: dict[str, Any] | None = None, 
                    batch_size: int = 1, shuffle: bool = False, transform_func: A.Compose | None = None, desire_bbox_format: str = 'yolo', pin_memory: bool = False):
    dataset = LoadDataset(
        dataset_dir_path=Path(root_dir),
        coco_annotations_path=Path(coco_path),
        set_ratio=set_ratio,
        transforms=transform_func,
        desire_bbox_format=desire_bbox_format
    )
    if custom_categories is not None:
        dataset.load_categories(custom_categories=custom_categories)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )

    return dataloader