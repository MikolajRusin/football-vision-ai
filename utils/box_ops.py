import torch
import numpy as np

def normalize_bboxes(bboxes: np.ndarray | torch.Tensor, image_height: int | float, image_width: int | float):
    if bboxes is None or len(bboxes) == 0:
        if isinstance(bboxes, torch.Tensor):
            return torch.zeros((0, 4), dtype=torch.float32)
        return np.zeros((0, 4), dtype=np.float32)

    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.clone().float()
    else:
        bboxes = bboxes.astype(np.float32)

    bboxes[:, [0, 2]] /= float(image_width)
    bboxes[:, [1, 3]] /= float(image_height)

    bboxes = torch.clamp(bboxes, 0, 1) if isinstance(bboxes, torch.Tensor) else np.clip(bboxes, 0, 1)
    return bboxes