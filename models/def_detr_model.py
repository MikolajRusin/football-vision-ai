from transformers import AutoImageProcessor, DeformableDetrConfig, DeformableDetrObjectDetection
import torch
import torch.nn as nn

class DefDetrModel(nn.Module):
    def __init__(self, model_id: str, id2label: dict[int, str] | None = None, device: str | None = None):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_id = model_id
        self.id2label = id2label

        # --- load config and model
        self.config = self._load_model_config()
        self.model = self._load_model().to(device)
        self.processor = self.AutoImageProcessor.from_pretrained(self.model_id, use_fast=True)

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

    def forward(self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]] | None = None):
        inputs = self.processor(images=images, return_tensors='pt', device=self.device)

        if targets is not None:
            outputs = self.model(**inputs, labels=targets)
        else:
            outputs = self.model(**inputs)
        return outputs