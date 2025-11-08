import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utils.box_ops import denormalize_bboxes
from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.data import DataLoader
from logger.wandb_logger import WandbLogger
from transformers.modeling_outputs import BaseModelOutput
from evaluator import Evaluator
import gc

@dataclass
class TransformerTrainer:
    model: nn.Module
    train_dataloader: DataLoader
    valid_dataloader: DataLoader | None = None
    val_frequency: int | None = None
    wandb_logger: WandbLogger | None = None
    map_per_class: bool = False
    n_epochs: int = 5
    optimizer: str = 'adamw'
    optimizer_params: dict | None = None
    lr_scheduler: str | None = None
    lr_scheduler_params: dict | None = None

    def __post_init__(self):
        # Set model type
        self.model_type = self._get_model_type()

        # Set optimizer and, if lr_scheduler is specified also lr_scheduler
        self.optimizer, self.lr_scheduler = self._configure_optimizer_and_scheduler()
        # Initialize evaluator to calculate metrics
        self.evaluator = Evaluator(map_per_class=self.map_per_class)

        # Lists for tracking loss
        self.train_loss = []
        self.valid_loss = []

    def train(self):
        print(' Start Training '.center(90, '-'))
        self._display_training_params()

        for cur_n_epoch in range(self.n_epochs):
            # Train
            train_loss = self._train_one_epoch(cur_n_epoch)
            self.train_loss.append(train_loss)   

            # Evaluate
            if self.valid_dataloader is not None:
                valid_loss = self.evaluate_model(self.train_dataloader)
                self.valid_loss.append(valid_loss)

            # Update learning rate if provided scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        print(' End Training '.center(90, '-'))

    def _train_one_epoch(self, epoch):
        running_loss = 0
        running_box_loss = 0
        running_total_samples = 0

        # Iteration loop for training one epoch
        for iteration, (batch_image, batch_target) in enumerate(tqdm(self.train_dataloader, total=len(self.train_dataloader), desc=f'Epoch {epoch}/{self.n_epochs}')):
            torch.cuda.empty_cache()  # Clear the CUDA cache
            self.model.train()        # Set the model to train mode

            # Get the model's predictions for a batch, calculate losses and make the common steps
            outputs = self.model(batch_image, batch_target)
            loss = outputs.loss
            self._common_steps(loss)

            print('pred_boxespred_boxespred_boxespred_boxespred_boxes')
            print(type(outputs))
            print(outputs.pred_boxes)
            print('\nlogitslogitslogitslogitslogitslogitslogits')
            print(outputs.logits.argmax(dim=-1))
            print('\nloss')
            print(outputs.loss)
            print('\nkeykeykeykeykeykeykey')
            print(outputs.keys())
            break
            
            # Add loss to the total loss for current epoch
            running_loss += loss
            running_total_samples += len(batch_image)

            # Model validation with specified val_frequency
            if self.val_frequency is not None and ((iteration + 1) % self.val_frequency) == 0:
                freq_val_loss  =  self.evaluate_model(self.valid_dataloader)
                avg_train_loss = running_loss / running_total_samples

        return running_loss / len(self.train_dataloader)
    
    @torch.no_grad()
    def evaluate_model(self, dataloader: DataLoader):
        print(' Start Evaluating '.center(90, '-'))
        self.model.eval()  # Set the model to valid mode

        running_loss = 0
        all_preds = []
        all_targets = []
        
        # Iteration loop for model validation
        for (batch_image, batch_target) in tqdm(dataloader, total=len(dataloader), desc='Evaluating model'):
            outputs = self.model(batch_image, batch_target)
            loss = outputs.loss
            running_loss += loss

            # if self.wandb_logger is not None:
            preds    = self._postprocess_outputs_for_evaluator(outputs)
            targets  = self._postprocess_targets_for_evaluator(batch_target)

            all_preds.extend(preds)
            all_targets.extend(targets)
        
        map_metrics = self.evaluator.compute_metrics(all_preds, all_targets)

        print('mapmapmapmapmapmapmapmapmapmapmapmapmapmapmapmapmapmapmap')
        print(map_metrics)

        self.model.train()
        return running_loss / len(dataloader)

    def _common_steps(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()   

    def _postprocess_targets_for_evaluator(self, batch_target: list[dict[str, torch.Tensor]]):
        targets_for_evaluator = [
            {
                'boxes': torch.stack([img_ann['bbox'] for img_ann in img_target['annotations']]),
                'labels': torch.tensor([img_ann['category_id'] for img_ann in img_target['annotations']])
            }
            for img_target in batch_target
        ]
        print('boxesboxesboxesboxesboxesboxesboxesboxesboxesboxesboxesboxes')
        print([t['boxes'] for t in targets_for_evaluator])
        return targets_for_evaluator  
    
    def _postprocess_outputs_for_evaluator(self, outputs: BaseModelOutput):
        raw_logits      = outputs.logits
        raw_pred_boxes  = outputs.pred_boxes
        resized_sizes   = outputs.size
        orig_sizes      = outputs.orig_size
        
        model_type = self.model.__class__.__name__

        # Convert logits into scores and predicted class_ids
        preds = raw_logits.softmax(-1)
        scores, cls_ids = preds.max(-1)

        # Filter classes where cls_id != 0 (Background)
        score_threshold = 0.4
        keep_mask  = ((cls_ids != 0) & (scores >= score_threshold))
        print('scorescorescorescorescorescorescorescorescorescorescorescore')
        print(scores)
        print('sumsumsumsumsumsumsumsumsumsumsumsumsumsumsum')
        print(torch.sum(keep_mask))
        scores     = scores[keep_mask]
        cls_ids    = cls_ids[keep_mask]
        pred_boxes = raw_pred_boxes[keep_mask]

        # Add batch dimension for single image (if needed)
        scores     = scores.unsqueeze(0) if scores.ndim == 1 else scores
        cls_ids    = cls_ids.unsqueeze(0) if cls_ids.ndim == 1 else cls_ids
        pred_boxes = pred_boxes.unsqueeze(0) if pred_boxes.ndim == 2 else pred_boxes
        

        # Combine all data
        preds_for_evaluator = [
            {
                'boxes': self._convert_pred_boxes(
                    pred_boxes=img_pred_boxes, 
                    resized_size=tuple(resized_size.tolist()),
                    target_size=tuple(orig_size.tolist())
                ).detach().cpu(),
                'scores': img_scores.detach().cpu(),
                'labels': img_cls_ids.detach().cpu()
            }
            for img_pred_boxes, img_scores, img_cls_ids, resized_size, orig_size
            in zip(pred_boxes, scores, cls_ids, resized_sizes, orig_sizes)
        ]
        print('pred_bboxespred_bboxespred_bboxespred_bboxespred_bboxespred_bboxespred_bboxes')
        print([t['boxes'] for t in preds_for_evaluator])
        return preds_for_evaluator

    def _convert_pred_boxes(self, pred_boxes: torch.Tensor, resized_size: tuple[int, int], target_size: tuple[int, int]):
        if self.model_type == 'detr':
            pass

    def _configure_optimizer_and_scheduler(self) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
        # Configure Optimizer
        if getattr(self.optimizer_params, 'backbone_lr', None):
            params_dict = [
                {'params': [p for n, p in self.model.named_parameters() if 'backbone' in n and p.requires_grad], 'lr': self.optimizer_params['backbone_lr']},
                {'params': [p for n, p in self.model.named_parameters() if 'backbone' not in n and p.requires_grad]}
            ]
        else:
            params_dict = [
                {'params': [p for _, p in self.model.named_parameters() if p.requires_grad]}
            ]
        backbone_lr = self.optimizer_params.pop('backbone_lr', None)  # Temporary deletion of 'backbone_lr' before passing the parameters to optimizer
        optimizer = self._get_optimizer_cls(self.optimizer)(params_dict, **self.optimizer_params)
        self.optimizer_params['backbone_lr'] = backbone_lr  # Add previously deleted 'backbone_lr'

        # Configure Scheduler
        scheduler = None
        if self.lr_scheduler is not None:
            scheduler = self._get_scheduler_cls(self.lr_scheduler)(optimizer, **self.lr_scheduler_params)

        return optimizer, scheduler

    def _get_model_type(self):
        model_type = self.model.__class__.__name__.lower()
        if 'deta' in self.model_type:
            model_type = 'deta'
        elif 'detr' in self.model_type:
            model_type = 'detr'
        return model_type

    def _get_optimizer_cls(self, optimizer_name: str) -> torch.optim.Optimizer:
        optimizers_cls = {
            'adamw': optim.AdamW,
            'adam': optim.Adam,
            'sgd': optim.SGD
        }
        return optimizers_cls[optimizer_name.lower()]

    def _get_scheduler_cls(self, scheduler_name: str) -> torch.optim.lr_scheduler.LRScheduler:
        schedulers_cls = {
            'cosine_annealing': lr_scheduler.CosineAnnealingLR,
            'step': lr_scheduler.StepLR,
            'onecycle': lr_scheduler.OneCycleLR
        }
        return schedulers_cls[scheduler_name.lower()]

    def _display_training_params(self):
        print(' Training Params '.center(90, '-'))
        print(f'Training model: {self.model.__class__.__name__}')
        print(f'The model has been loaded on {'cuda' if next(self.model.parameters()).is_cuda else 'cpu'}')
        print(f'Number of images in train dataloader: {len(self.train_dataloader)}')
        if self.valid_dataloader is not None:
            print(f'Number of images in valid dataloader: {len(self.valid_dataloader)}')
        print(f'Number of training epochs: {self.n_epochs}')
        if self.val_frequency is not None:
            print(f'Frequency of evaluating model: {self.val_frequency}')
        print(f'Used optimizer: {self.optimizer.__class__.__name__}')
        print(f"Optimizer's params: {self.optimizer_params}")
        if self.lr_scheduler is not None:
            print(f'Used learning rate scheduler: {self.lr_scheduler}')
            print(f"Learning rate scheduler's params: {self.lr_scheduler_params}")
        print(f"Log metrics: {'False' if self.wandb_logger is None else 'True'}")
        if self.wandb_logger is not None:
            print(f'Computer mAP per class: {self.map_per_class}')
        print('\n')