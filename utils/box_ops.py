import torch
import numpy as np

def clip_bboxes_range(bboxes: np.ndarray | torch.Tensor, height: int | float, width: int | float):
    bboxes[: [0, 2]] = torch.clamp(bboxes[:, [0, 2]], 0.0, float(width)) if isinstance(bboxes, torch.Tensor) else np.clip(bboxes[:, [0, 2]], 0.0, float(width))
    bboxes[: [1, 3]] = torch.clamp(bboxes[:, [1, 3]], 0.0, float(height)) if isinstance(bboxes, torch.Tensor) else np.clip(bboxes[:, [1, 3]], 0.0, float(height))
    return bboxes

def resize_bboxes(bboxes: np.ndarray | torch.Tensor, size: tuple[int, int], target_size: tuple[int, int]):
    h, w = size
    target_h, target_w = target_size
    scale_x = target_w / w
    scale_y = target_h / h

    if isinstance(bboxes, torch.Tensor):
        resized_bboxes = bboxes.clone().float()
    else:
        resized_bboxes = bboxes.astype(np.float32)

    resized_bboxes[:, [0, 2]] = resized_bboxes[:, [0, 2]] * scale_x
    resized_bboxes[:, [1, 3]] = resized_bboxes[:, [1, 3]] * scale_y
    resized_bboxes = clip_bboxes_range(resized_bboxes, target_h, target_w)
    return resized_bboxes

def normalize_area(area: np.ndarray | torch.Tensor, image_height: int | float, image_width: int | float):
    area = area / (image_height * image_width)
    return area

def denormalize_bboxes(bboxes: np.ndarray | torch.Tensor, image_height: int | float, image_width: int | float):
    if bboxes is None or len(bboxes) == 0:
        if isinstance(bboxes, torch.Tensor):
            return torch.zeros((0, 4), dtype=torch.float32)
        return np.zeros((0, 4), dtype=np.float32)

    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.clone().float()
    else:
        bboxes = bboxes.astype(np.float32)
    
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * float(image_width)
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * float(image_height)
    bboxes = clip_bboxes_range(bboxes, image_height, image_width)
    return bboxes

def normalize_bboxes(bboxes: np.ndarray | torch.Tensor, image_height: int | float, image_width: int | float):
    if bboxes is None or len(bboxes) == 0:
        if isinstance(bboxes, torch.Tensor):
            return torch.zeros((0, 4), dtype=torch.float32)
        return np.zeros((0, 4), dtype=np.float32)

    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.clone().float()
        if torch.all((bboxes >= 0.0) & (bboxes <= 1.0)):
            return bboxes
    else:
        bboxes = bboxes.astype(np.float32)
        if np.all(((bboxes >= 0) & (bboxes <= 1))):
            return bboxes

    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / float(image_width)
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / float(image_height)

    bboxes = clip_bboxes_range(bboxes, 1.0, 1.0)
    return bboxes

def xywh2xcycwh(bboxes: np.ndarray | torch.Tensor):
    if bboxes is None or len(bboxes) == 0:
        if isinstance(bboxes, torch.Tensor):
            return torch.zeros((0, 4), dtype=torch.float32)
        return np.zeros((0, 4), dtype=np.float32)
    
    bboxes = bboxes.clone().float() if isinstance(bboxes, torch.Tensor) else bboxes.astype(np.float32)

    bboxes[:, 0] = bboxes[:, 0] + (bboxes[:, 2] / 2)
    bboxes[:, 1] = bboxes[:, 1] + (bboxes[:, 3] / 2)
    return bboxes
    
def xywh2xyxy(bboxes: np.ndarray | torch.Tensor, image_height: int | float, image_width: int | float):
    if bboxes is None or len(bboxes) == 0:
        if isinstance(bboxes, torch.Tensor):
            return torch.zeros((0, 4), dtype=torch.float32)
        return np.zeros((0, 4), dtype=np.float32)

    is_tensor = isinstance(bboxes, torch.Tensor)
    bboxes = bboxes.clone().float() if is_tensor else bboxes.astype(np.float32)

    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    bboxes = normalize_bboxes(bboxes, image_height, image_width)
    return bboxes

def cxcywh2xywh(bboxes: np.ndarray | torch.Tensor, image_height: int | float, image_width: int | float):
    pass