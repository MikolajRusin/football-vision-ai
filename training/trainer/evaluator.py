from dataclasses import dataclass
from transformers.modeling_outputs import BaseModelOutput
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch

@dataclass
class Evaluator:
    map_per_class: bool = False

    def __post_init__(self):
        self.map_metric = MeanAveragePrecision(iou_type='bbox', box_format='xywh', class_metrics=self.map_per_class)

    def compute_metrics(self, preds: list[dict[str: torch.Tensor]], targets: list[dict[str, torch.Tensor]]):
        self.map_metric.update(preds, targets)
        map_results = self.map_metric.compute()
        self.map_metric.reset()
        return map_results