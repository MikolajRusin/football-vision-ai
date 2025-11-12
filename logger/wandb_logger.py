from curses import has_key
from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from typing import Any
import wandb

@dataclass
class WandbLogger:
    project_name: str
    project_config: dict[Any] | None = None

    def __post_init__(self):
        self.run = wandb.init(
            project=self.project_name,
            config=self.project_config
        )

    def log_results(self, results: dict[str, dict[str, float]], stage: str='epoch'):
        losses = {f'{stage}/{k}': v for k, v in results['losses'].items()} if 'losses' in results else {}
        metrics = {f'{stage}/{k}': v for k, v in results['metrics'].items()} if 'metrics' in results else {}

        if losses or metrics:
            self.run.log({
                **losses,
                **metrics
            })
