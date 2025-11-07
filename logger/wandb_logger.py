from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision

@dataclass
class WandbLogger:
    map_per_class=False

    def __post_init__(self):
        map_metrics = MeanAveragePrecision(class_metrics=self.map_per_class)