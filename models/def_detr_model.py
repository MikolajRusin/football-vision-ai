from transformers import DeformableDetrImageProcessor, DeformableDetrConfig, DeformableDetrForObjectDetection
from transformers.modeling_outputs import BaseModelOutput
import torch
import torch.nn as nn
from typing import Any

class DefDetrModel(nn.Module):
    def __init__(self, model_id: str, id2label: dict[int, str] | None = None, device: str | None = None):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_id = model_id
        self.id2label = id2label

        # --- load config and model
        self.config = self._load_model_config()
        self.model = self._load_model().to(self.device)
        self.processor = DeformableDetrImageProcessor.from_pretrained(self.model_id, use_fast=True)

    def _load_model_config(self):
        config = DeformableDetrConfig.from_pretrained(self.model_id)
        if self.id2label is not None:
            config.id2label   = self.id2label
            config.label2id   = {v: k for k, v in self.id2label.items()}
            config.num_labels = len(self.id2label)
        return config

    def _reset_class_embeddings(self, model):
        for i in range(len(model.class_embed)):
            nn.init.xavier_uniform(model.class_embed[i].weight)
            if model.class_embed[i].bias is not None:
                nn.init.constant_(model.class_embed[i].bias, 0)

    def _load_model(self) -> DeformableDetrForObjectDetection:
        model = DeformableDetrForObjectDetection.from_pretrained(self.model_id, config=self.config, ignore_mismatched_sizes=True)
        if self.id2label is not None:
            self._reset_class_embeddings(model)
        return model

    # @torch.no_grad()
    def forward(self, images: list[torch.Tensor], targets: list[dict[str, Any]] | None = None) -> BaseModelOutput:
        if targets is not None:
            inputs = self.processor(images=images, annotations=targets, return_tensors='pt')
            pixel_values = inputs['pixel_values'].to(self.device)
            pixel_mask   = inputs['pixel_mask'].to(self.device)
            labels       = [{k: v.to(self.device) for k, v in l.items()} for l in inputs['labels']]
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        else:
            inputs = self.processor(images=images, return_tensors='pt')
            pixel_values = inputs['pixel_values'].to(self.device)
            pixel_mask   = inputs['pixel_mask'].to(self.device)
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs