import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler

@dataclass
class TransformerTrainer:
    model: nn.Module
    train_dataloader: DataLoader
    valid_dataloader: DataLoader | None = None
    n_epochs: int = 5
    optimizer_cls: torch.optim.Optimizer = AdamW
    optimizer_additional_params: dict | None = None
    lr_scheduler_cls: torch.optim.lr_scheduler.LRScheduler | None = None
    lr_scheduler_params: dict | None = None
    lr: float = 1e-4
    lr_backbone: float | None = None
    weight_decay: float | None = None

    def __post_init__(self):
        self.optimizer, self.lr_scheduler = self._configure_optimizers()

    def train(self):
        for cur_n_epoch in range(self.n_epochs):
            pass

    def _common_steps(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _configure_optimizers(self) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
        # Param Groups
        if self.backbone_lr is not None:
            param_dicts = [
                {'params': [p for n, p in self.model.named_parameters if 'backbone' in n and p.requires_grad], 'lr': self.backbone_lr},
                {'params': [p for n, p in self.model.named_parameters if 'backbone' not in n and p.requires_grad]}
            ]
        else:
            param_dicts = [
                {'params': [p for _, p in self.model.named_parameters() if p.requires_grad]}
            ]

        # Optimizer
        optimizer_kwargs = {
            'lr': self.lr, 
            'weight_decay': self.weight_decay if self.weight_decay is not None else 0.01  # If self.weight_decay is None then use default values
        }
        if self.optimizer_additional_params is not None:
            self.optimizer_additional_params.pop('lr', None)
            optimizer_kwargs.update(self.optimizer_additional_params)
        optimizer = self.optimizer_cls(param_dicts, **optimizer_kwargs)

        # Learning Rate Scheduler
        lr_scheduler = None
        if self.lr_scheduler_cls is not None:
            scheduler_kwargs = {'optimizer': optimizer}
            if self.lr_scheduler_cls is not None:
                scheduler_kwargs.update(self.lr_scheduler_cls)
            lr_scheduler = self.lr_scheduler_cls(**scheduler_kwargs)
        
        return optimizer, lr_scheduler