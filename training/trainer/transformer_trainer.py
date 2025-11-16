import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utils.box_ops import cxcywh2xywh, denormalize_bboxes, resize_bboxes
from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.data import DataLoader
from logger.wandb_logger import WandbLogger
from transformers.modeling_outputs import BaseModelOutput
from training.trainer.evaluator import Evaluator
from manager.checkpoint_manager import ModelCheckpointManager

@dataclass
class TransformerTrainer:
    model: nn.Module
    train_dataloader: DataLoader
    valid_dataloader: DataLoader | None = None
    frequency_validating: int | None = None
    n_epochs: int = 5
    optimizer: str = 'adamw'
    optimizer_params: dict | None = None
    lr_scheduler: str | None = None
    lr_scheduler_params: dict | None = None
    checkpoint_manager: ModelCheckpointManager | None = None
    frequency_saving_checkpoint: int | None = None
    wandb_logger: WandbLogger | None = None
    map_per_class: bool = False

    def __post_init__(self):
        # Set model type
        self.model_type = self._get_model_type()

        # Set optimizer and, if lr_scheduler is specified also lr_scheduler
        self.optimizer, self.lr_scheduler = self._configure_optimizer_and_scheduler()
        # Initialize evaluator to calculate metrics
        self.evaluator = Evaluator(
            map_per_class=self.map_per_class, 
            id2label=self.model.id2label if hasattr(self.model, 'id2label') else None
        )

        # Lists for tracking loss
        self.train_loss = []
        self.valid_loss = []
        self.bbox_loss  = []

    def train(self) -> None:
        print(' Start Training '.center(90, '-'))
        self._display_training_params()

        for epoch in range(self.n_epochs):
            cur_n_epoch = epoch + 1
            epoch_results = {
                'losses': {}
            }

            # Train
            train_results = self._train_one_epoch(cur_n_epoch)
            train_loss = train_results['losses']['loss']
            train_bbox_loss = train_results['losses']['bbox_loss']
            epoch_results['losses']['train_loss'] = float(train_loss)
            epoch_results['losses']['train_bbox_loss'] = float(train_bbox_loss)
            self.train_loss.append(train_loss)
            self.bbox_loss.append(train_bbox_loss)

            # Evaluate
            if self.valid_dataloader is not None:
                valid_results = self.evaluate_model(self.valid_dataloader)
                valid_loss = valid_results['losses']['loss']
                valid_bbox_loss = valid_results['losses']['bbox_loss']
                valid_metrics = valid_results['metrics']
                epoch_results['losses']['valid_loss'] = float(valid_loss)
                epoch_results['losses']['valid_bbox_loss'] = float(valid_bbox_loss)
                epoch_results['metrics'] = {k: float(v) for k, v in valid_metrics.items()}
                self.valid_loss.append(valid_loss)

            # Update learning rate if provided scheduler
            if self.lr_scheduler is not None:
                epoch_results['learning_rates'] = {}
                curr_learning_rates = self.lr_scheduler.get_last_lr()
                if len(curr_learning_rates) == 2:
                    epoch_results['learning_rates']['backbone_lr'] = float(curr_learning_rates[0])
                    epoch_results['learning_rates']['lr'] = float(curr_learning_rates[1])
                else:
                    epoch_results['learning_rates']['lr'] = float(curr_learning_rates[0])
                self.lr_scheduler.step()

            # Log epoch results to wandb
            if self.wandb_logger is not None:
                self.wandb_logger.log_results(epoch_results, stage='epoch')

            # Save epoch model's checkpoint
            if self.checkpoint_manager is not None:
                self.checkpoint_manager.save_checkpoint(self.model, epoch=cur_n_epoch)

        print(' End Training '.center(90, '-'))

    def _train_one_epoch(self, cur_n_epoch) -> dict[str, float]:
        running_loss = 0
        running_bbox_loss = 0
        running_total_samples = 0

        # Iteration loop for training one epoch
        for iteration, (batch_image, batch_target) in enumerate(tqdm(self.train_dataloader, total=len(self.train_dataloader), desc=f'Epoch {cur_n_epoch}/{self.n_epochs}')):
            torch.cuda.empty_cache()  # Clear the CUDA cache
            self.model.train()        # Set the model to train mode
            cur_n_iteration = iteration + 1
            iteration_results = {
                'losses': {}
            }

            # Get the model's predictions for a batch, calculate losses and make the common steps
            outputs = self.model(batch_image, batch_target)
            loss = outputs.loss
            self._common_steps(loss)

            # Add loss to the total runnning_loss for current epoch
            iteration_loss = loss.detach().cpu().item()
            running_loss += iteration_loss

            # Add bbox loss to the running_bbox_loss
            iteration_bbox_loss = outputs.loss_dict['loss_bbox'].detach().cpu().item()
            running_bbox_loss += iteration_bbox_loss

            # Add length of batch for further calculations
            running_total_samples += len(batch_image)

            # Log ireation results to wandb
            if self.wandb_logger is not None:
                iteration_results['losses']['loss'] = float(iteration_loss)
                iteration_results['losses']['bbox_loss'] = float(iteration_bbox_loss)
                self.wandb_logger.log_results(iteration_results, stage='iteration')
                            
            # Model validation with specified frequency_validating
            if self.frequency_validating is not None and (cur_n_iteration % self.frequency_validating) == 0:
                freq_results = {
                    'losses': {}
                }
                # Frequency validation results
                freq_val_results = self.evaluate_model(self.valid_dataloader)
                freq_metrics = freq_val_results['metrics']
                freq_val_loss = freq_val_results['losses']['loss']
                freq_val_bbox_loss = freq_val_results['losses']['bbox_loss']
                # Add frequency validation results to the freq_results 
                freq_results['metrics'] = {k: float(v) for k, v in freq_metrics.items()}
                freq_results['losses']['val_loss'] = float(freq_val_loss)
                freq_results['losses']['val_bbox_loss'] = float(freq_val_bbox_loss)

                # Frequency train results
                freq_train_loss = running_loss / running_total_samples
                freq_train_bbox_loss = running_bbox_loss / running_total_samples
                # Add frequency train results to the freq_results 
                freq_results['losses']['train_loss'] = float(freq_train_loss)
                freq_results['losses']['train_bbox_loss'] = float(freq_train_bbox_loss)

                # Log frequency results to wandb
                if self.wandb_logger is not None:
                    self.wandb_logger.log_results(freq_results, stage=f'frequency_{self.frequency_validating}')

            # Save frequency model's checkpoint
            if (self.frequency_saving_checkpoint is not None and self.checkpoint_manager is not None)\
                and (cur_n_iteration % self.frequency_saving_checkpoint) == 0:
                self.checkpoint_manager.save_checkpoint(self.model, epoch=cur_n_epoch, iteration=cur_n_iteration)

            avg_train_loss = running_loss / running_total_samples
            avg_train_bbox_loss = running_bbox_loss / running_total_samples
        return {
            'losses': {
                'loss': avg_train_loss,
                'bbox_loss': avg_train_bbox_loss
            }
        }
    
    @torch.no_grad()
    def evaluate_model(self, dataloader: DataLoader):
        print(' Start Evaluating '.center(90, '-'))
        self.model.eval()  # Set the model to valid mode

        running_loss = 0
        running_bbox_loss = 0
        all_preds = []
        all_targets = []
        
        # Iteration loop for model validation
        for (batch_image, batch_target) in tqdm(dataloader, total=len(dataloader), desc='Evaluating model'):
            outputs = self.model(batch_image, batch_target)
            # Validation loss
            loss = outputs.loss
            running_loss += loss.cpu().item()
            # Validation bbox loss
            bbox_loss = outputs.loss_dict['loss_bbox']
            running_bbox_loss += bbox_loss.cpu().item()

            preds    = self._postprocess_outputs_for_evaluator(outputs)
            targets  = self._postprocess_targets_for_evaluator(batch_target)

            all_preds.extend(preds)
            all_targets.extend(targets)
        
        map_metrics         = self.evaluator.compute_metrics(all_preds, all_targets)
        avg_valid_loss      = running_loss / len(dataloader)
        avg_valid_bbox_loss = running_bbox_loss / len(dataloader)

        self.model.train()
        return {
            'losses': {
                'loss': avg_valid_loss,
                'bbox_loss': avg_valid_bbox_loss
            },
            'metrics': map_metrics
        }

    def _common_steps(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()   

    def _postprocess_targets_for_evaluator(self, batch_target: list[dict[str, torch.Tensor]]):
        targets_for_evaluator = [
            {
                'boxes': torch.stack([img_ann['bbox'] for img_ann in img_target['annotations']]).detach().cpu(),
                'labels': torch.tensor([img_ann['category_id'] for img_ann in img_target['annotations']]).detach().cpu()
            }
            for img_target in batch_target
        ]
        return targets_for_evaluator  
    
    def _postprocess_outputs_for_evaluator(self, outputs: BaseModelOutput):
        raw_logits      = outputs.logits
        raw_pred_boxes  = outputs.pred_boxes
        resized_sizes   = outputs.size
        orig_sizes      = outputs.orig_size

        preds = raw_logits.softmax(-1)
        scores, cls_ids = preds.max(-1)

        score_threshold = 0.3

        batch_boxes  = []
        batch_scores = []
        batch_labels = []

        for i in range(raw_pred_boxes.shape[0]):
            keep_mask = (cls_ids[i] != 0) & (scores[i] >= score_threshold)
            batch_boxes.append(raw_pred_boxes[i][keep_mask])
            batch_scores.append(scores[i][keep_mask])
            batch_labels.append(cls_ids[i][keep_mask])

        preds_for_evaluator = [
            {
                'boxes': self._convert_pred_boxes_for_evaluator(
                    pred_boxes=img_pred_boxes, 
                    resized_size=tuple(resized_size.tolist()),
                    target_size=tuple(orig_size.tolist())
                ).detach().cpu(),
                'scores': img_scores.detach().cpu(),
                'labels': img_cls_ids.detach().cpu()
            }
            for img_pred_boxes, img_scores, img_cls_ids, resized_size, orig_size
            in zip(batch_boxes, batch_scores, batch_labels, resized_sizes, orig_sizes)
        ]

        return preds_for_evaluator

    def _convert_pred_boxes_for_evaluator(self, pred_boxes: torch.Tensor, resized_size: tuple[int, int], target_size: tuple[int, int]):
        if self.model_type in ['detr', 'deta']:
            pred_boxes = cxcywh2xywh(pred_boxes)
            pred_boxes = denormalize_bboxes(pred_boxes, resized_size[0], resized_size[1])
            pred_boxes = resize_bboxes(pred_boxes, resized_size, target_size)
        return pred_boxes 

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
        if 'deta' in model_type:
            model_type = 'deta'
        elif 'detr' in model_type:
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
        print(f'Model type: {self.model_type}')
        print(f'The model has been loaded on {'cuda' if next(self.model.parameters()).is_cuda else 'cpu'}')
        print(f'Number of images in train dataloader: {len(self.train_dataloader)}')
        if self.valid_dataloader is not None:
            print(f'Number of images in valid dataloader: {len(self.valid_dataloader)}')
        print(f'Number of training epochs: {self.n_epochs}')
        if self.frequency_validating is not None:
            print(f'Frequency of evaluating model: {self.frequency_validating}')
        print(f'Used optimizer: {self.optimizer.__class__.__name__}')
        print(f"Optimizer's params: {self.optimizer_params}")
        if self.lr_scheduler is not None:
            print(f'Used learning rate scheduler: {self.lr_scheduler}')
            print(f"Learning rate scheduler's params: {self.lr_scheduler_params}")
        print(f"Log metrics: {'False' if self.wandb_logger is None else 'True'}")
        if self.wandb_logger is not None:
            print(f'Computer mAP per class: {self.map_per_class}')
        print('\n')