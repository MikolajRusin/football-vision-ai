from utils.box_ops import xywh2cxcywh, xywh2xyxy, normalize_area, normalize_bboxes
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
    transforms: albumentations.Compose | None = None
    desire_bbox_format: str = 'xywh'
    return_img_path: bool = False

    def __post_init__(self):
        self.load_data()
        self.load_categories()
        self.changed_categories = False

    def load_data(self):
        self.coco = COCO(self.coco_annotations_path)
        self.ids  = list(self.coco.imgs.keys())

        if self.set_ratio is not None:
            self._filter_indices_by_ratio()

    def load_categories(self, custom_categories: list[dict[Any, Any]] | None = None):
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        if custom_categories is not None:
            if len(self.categories) != len(custom_categories):
                raise ValueError('Old categories and custom categories must be the same length')
            else:
                self.old_categories = self.categories
                self.categories = custom_categories
                self.old2new_id = {old_id: new_id for old_id, new_id in zip([d['id'] for d in self.old_categories], [d['id'] for d in self.categories])}
                
                print(f'Changed {self.old_categories} categories to {self.categories} categories')
                self.changed_categories = True


    def _filter_indices_by_ratio(self):
        n_ids = len(self.ids)

        if isinstance(self.set_ratio, float):
            n_keep = int(n_ids * self.set_ratio)
        elif isinstance(self.set_ratio, int):
            n_keep = self.set_ratio

        self.ids = self.ids[:n_keep]

    def _load_image(self, id: int) -> tuple[np.ndarray, str]:
        img_filename = self.coco.loadImgs(id)[0]['file_name']
        img = cv2.imread(str(self.dataset_dir_path / img_filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, img_filename
    
    def _load_anns(self, id: int) -> tuple[list[float], list[int]]:
        anns         = self.coco.loadAnns(self.coco.getAnnIds(id))
        bboxes       = [ann['bbox'] for ann in anns]
        category_ids = [ann['category_id'] for ann in anns]
        areas        = [ann['area'] for ann in anns]
        if self.changed_categories:
            category_ids = [self.old2new_id[old_id] for old_id in category_ids]
        return bboxes, category_ids, areas
        
    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        id = self.ids[index]
        image, image_filename = self._load_image(id)
        bboxes, category_ids, areas = self._load_anns(id)

        # Albumentations
        if self.transforms is not None:
            # Convert to numpy array
            bboxes       = np.array(bboxes, dtype=np.float32)
            category_ids = np.array(category_ids, dtype=np.int64)
            # Albumentation by transforms function
            transformed  = self.transforms(image=image, bboxes=bboxes, category_ids=category_ids)
            image        = transformed['image']
            bboxes       = transformed['bboxes']
            category_ids = transformed['category_ids']
            # Recalculate area after transformation
            bboxes = np.array(bboxes, dtype=np.float32)
            width  = bboxes[:, 2]
            height = bboxes[:, 3]
            areas  = width * height

        # Ensure image is a tensor with uint8 dtype (0-255 range)
        # The processor will handle normalization
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image, dtype=torch.uint8)
        elif image.dtype != torch.uint8:
            # If already a tensor but not uint8, ensure it's in 0-255 range
            if image.dtype == torch.float32 and image.max() <= 1.0:
                image = (image * 255).to(torch.uint8)
            else:
                image = image.to(torch.uint8)
            
        # If image is not in CHW format (i.e., HWC), permute to CHW
        if image.shape[0] != 3:
            image = image.permute(2, 0, 1)

        # Convert bboxes and category_ids to tensors
        bboxes       = torch.as_tensor(bboxes, dtype=torch.float32)
        category_ids = torch.as_tensor(category_ids, dtype=torch.int64)
        areas        = torch.as_tensor(areas, dtype=torch.float32)

        # Convert bboxes if desire_bbox_format is different than coco 
        # The code for area normalization is duplicated for future updates
        if self.desire_bbox_format == 'cxcywh':
            bboxes = xywh2cxcywh(bboxes)
            bboxes = normalize_bboxes(bboxes, image_height=image.shape[1], image_width=image.shape[2])
            areas = normalize_area(areas, image_height=image.shape[1], image_width=image.shape[2])
        elif self.desire_bbox_format == 'xyxy':
            bboxes = xywh2xyxy(bboxes)
            bboxes = normalize_bboxes(bboxes, image_height=image.shape[1], image_width=image.shape[2])
            areas = normalize_area(areas, image_height=image.shape[1], image_width=image.shape[2])
        elif self.desire_bbox_format == 'xywh':
            # xywh format is the default format for saved dataset
            pass
        
        target = {
            'image_id': id,
            'annotations': [
                {
                    'bbox': bbox,
                    'category_id': cat_id,
                    'area': area
                }
                for bbox, cat_id, area in zip(bboxes, category_ids, areas)
            ]
        }
        if self.return_img_path:
            target['image_path'] = str(self.dataset_dir_path / image_filename)

        return image, target
