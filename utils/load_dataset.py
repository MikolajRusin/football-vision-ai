from torch.utils.data import Dataset
from pycocotools.coco import COCO
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import albumentations
import numpy as np
import torch
import cv2

@dataclass
class LoadDataset(Dataset):
    dataset_dir_path: Path
    coco_annotations_path: Path
    set_ratio: float | int | None = None
    transforms: albumentations.Compose = None

    def __post_init__(self):
        self.load_data()
        self._load_categories()

    def load_data(self):
        self.coco = COCO(self.coco_annotations_path)
        self.ids  = list(self.coco.imgs.keys())

        if self.set_ratio is not None:
            self._filter_indices_by_ratio()

    def _load_categories(self):
        self.categories = self.coco.loadCats(self.coco.getCatIds())

    def _filter_indices_by_ratio(self):
        n_ids = len(self.ids)

        if isinstance(self.set_ratio, float):
            n_keep = int(n_ids * self.set_ratio)
        elif isinstance(self.set_ratio, int):
            n_keep = self.set_ratio

        self.ids = self.ids[:n_keep]

    def _load_image(self, id: int) -> np.ndarray:
        img_filename = self.coco.loadImgs(id)[0]['file_name']
        img = cv2.imread(str(self.dataset_dir_path / img_filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _load_anns(self, id: int) -> tuple[list[float], list[int]]:
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        bboxes = [ann['bbox'] for ann in anns]
        category_ids = [ann['category_id'] for ann in anns]
        return bboxes, category_ids
        
    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        id = self.ids[index]
        image = self._load_image(id)
        bboxes, category_ids = self._load_anns(id)

        # Albumentatins
        if self.transforms is not None:
            # Convert to numpy array
            bboxes = np.array(bboxes, dtype=np.float32)
            category_ids = np.array(category_ids, dtype=np.int64)
            # Albumentation by transforms function
            transformed  = self.transforms(image=image, bboxes=bboxes, category_ids=category_ids)
            image        = transformed['image']
            bboxes       = transformed['bboxes']
            category_ids = transformed['category_ids']

        # Ensure image is a tensor
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image, dtype=torch.float32)

        # If image is not in CHW format (i.e., HWC), permute to CHW
        if image.shape[0] != 3:
            image = image.permute(2, 0, 1)

        # Convert bboxes and category_ids to tensors
        bboxes       = torch.as_tensor(bboxes, dtype=torch.float32)
        category_ids = torch.as_tensor(category_ids, dtype=torch.int64)

        target = {
            'bboxes': bboxes,
            'labels': category_ids
        }

        return image, target
