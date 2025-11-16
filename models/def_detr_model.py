from transformers import DeformableDetrImageProcessor, DeformableDetrConfig, DeformableDetrForObjectDetection
from transformers.modeling_outputs import BaseModelOutput
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

class DefDetrModel(nn.Module):
    def __init__(self, model_id: str, id2label: dict[int, str] | None = None, device: str | None = None, reset_head: bool = False):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_id = model_id
        self.id2label = id2label
        self.reset_head = reset_head

        # --- load config and model
        self.config = self._load_model_config()
        self.model = self._load_model(reset_head=self.reset_head).to(self.device)
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
            nn.init.xavier_uniform_(model.class_embed[i].weight)
            if model.class_embed[i].bias is not None:
                nn.init.constant_(model.class_embed[i].bias, 0)

    def _load_model(self, reset_head: bool) -> DeformableDetrForObjectDetection:
        model = DeformableDetrForObjectDetection.from_pretrained(
            self.model_id, 
            config=self.config, 
            ignore_mismatched_sizes=True
        )

        if reset_head:
            self._reset_class_embeddings(model)

        return model

    def load_model_checkpoint(self, checkpoint_path: str | Path):
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        config     = checkpoint['config']
        state_dict = checkpoint['model_state_dict']

        if isinstance(config, dict):
            config = DeformableDetrConfig(**config)
        elif not isinstance(config, DeformableDetrConfig):
            from omegaconf import OmegaConf
            config = DeformableDetrConfig(**OmegaConf.to_container(config, resolve=True))

        if self.config.num_labels != config.num_labels:
            self.config = config
            self.id2label = self.config.id2label
            self.model = self._load_model(reset_head=True).to(self.device)
        else:
            self.config = config
            self.model.config = config
        
        self.model.load_state_dict(state_dict)


    @classmethod
    def from_pretrained(cls, repo_id: str, device: str | None = None):
        local_dir = snapshot_download(repo_id=repo_id)

        processor = DeformableDetrImageProcessor.from_pretrained(local_dir)
        config = DeformableDetrConfig.from_pretrained(local_dir)

        model = cls(
            model_id=local_dir,
            id2label=config.id2label,
            device=device,
            reset_head=False
        )

        model.processor = processor

        weights_path = Path(local_dir) / 'model.safetensors'
        state_dict = load_file(str(weights_path))
        model.model.load_state_dict(state_dict)

        return model

    def forward(self, images: list[torch.Tensor], targets: list[dict[str, Any]] | None = None) -> BaseModelOutput:
        if targets is not None:
            inputs = self.processor(images=images, annotations=targets, return_tensors='pt')
            pixel_values = inputs['pixel_values'].to(self.device)
            pixel_mask   = inputs['pixel_mask'].to(self.device)
            labels       = [{k: v.to(self.device) for k, v in l.items()} for l in inputs['labels']]
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            
            outputs.size = [l['size'].to(self.device) for l in inputs['labels']]
            outputs.orig_size = [l['orig_size'].to(self.device) for l in inputs['labels']]
        else:
            with torch.no_grad():
                inputs = self.processor(images=images, return_tensors='pt')
                pixel_values = inputs['pixel_values'].to(self.device)
                pixel_mask   = inputs['pixel_mask'].to(self.device)
                outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
                
                resized_size = torch.tensor(pixel_values.shape[-2:], device=self.device)
                outputs.size = [resized_size for _ in range(len(images))]

                orig_size = [torch.tensor((img.shape[-2], img.shape[-1]), device=self.device) for img in images]
                outputs.orig_size = orig_size

        return outputs