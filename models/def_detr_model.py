from transformers import AutoImageProcessor, DeformableDetrConfig, DeformableDetrObjectDetection
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class DefDetrModel(nn.module):
    model_id: str
    id2label: dict[int, str] | None = None
    device: str = 'cpu'

    def __post_init__(self):
        super().__init__()
        self.config = self._load_model_config()
        self.model  = self._load_model()

    def _load_model_config(self):
        config = DeformableDetrConfig.from_pretrained(self.model_id)
        if self.id2label is not None:
            config.id2label   = self.id2label
            config.label2id   = {v: k for k, v in self.id2label.items()}
            config.num_labels = len(self.label2id)
        return config

    def _load_model(self):
        model = DeformableDetrObjectDetection.from_pretrained(self.model_id, config=self.config)
        # TODO: Add loading checkpoints
        return model

    def forward(self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]]):
        pass