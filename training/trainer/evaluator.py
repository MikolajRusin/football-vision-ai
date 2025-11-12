from dataclasses import dataclass
from transformers.modeling_outputs import BaseModelOutput
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch

@dataclass
class Evaluator:
    map_per_class: bool = False
    id2label: dict[id, str] | None = None

    def __post_init__(self):
        self.map_metric = MeanAveragePrecision(iou_type='bbox', box_format='xywh', class_metrics=self.map_per_class)

    def compute_metrics(self, preds: list[dict[str: torch.Tensor]], targets: list[dict[str, torch.Tensor]]):
        self.map_metric.update(preds, targets)
        map_results = self.map_metric.compute()
        map_results = self.postprocess_metrics(map_results)
        self.map_metric.reset()
        return map_results
    
    def postprocess_metrics(self, results):
        map_cls = {}
        if self.map_per_class is not None:
            map_cls = {
                f'mAP_cls_{self.id2label[int(cls_id)] if self.id2label is not None else int(cls_id)}': map_cls_result
                for map_cls_result, cls_id in zip(results['map_per_class'], results['classes'])
            }

        map_results = {
            'mAP_50-95': results['map'],
            'mAP_50': results['map_50'],
            'mAP_75': results['map_75'],
            'mAP_small': results['map_small'],
            'mAP_medium': results['map_medium'],
            'mAP_large': results['map_large'],
            **map_cls
        }
        return map_results