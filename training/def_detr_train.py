from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.data_utils.load_dataloader import load_dataloader
from logger.wandb_logger import WandbLogger
from models.def_detr_model import DefDetrModel
from training.transformer_trainer import TransformerTrainer
from dotenv import load_dotenv, find_dotenv
from omegaconf import OmegaConf
import albumentations as A
import argparse

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)
    return config

CONFIG = load_config()

# Paths of the project
PROJECT_PATH      = Path(CONFIG.paths.project_root).resolve()
TRAIN_IMAGES_PATH = Path(CONFIG.paths.train.images)
TRAIN_ANNOTATIONS = Path(CONFIG.paths.train.annotations)
VALID_IMAGES_PATH = Path(CONFIG.paths.valid.images) if getattr(CONFIG.paths.valid, 'images', None) else None
VALID_ANNOTATIONS = Path(CONFIG.paths.valid.annotations) if getattr(CONFIG.paths.valid, 'annotations', None) else None 

def main():
    load_dotenv(find_dotenv())

    # Augmentation function
    transforms = A.Compose(
        [
            A.HorizontalFlip(p=0.4),
            A.RandomBrightnessContrast(p=0.3),
            A.CLAHE(p=0.5),
            A.HueSaturationValue(p=0.3)
        ],
        bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'])
    )

    # Train and Valid dataloaders
    new_categories = [
        {'id': k, 'name': v}
        for k, v in CONFIG.model.id2label.items()
        if v != 'N/A'
    ]
    train_dataloader = load_dataloader(
        root_dir=TRAIN_IMAGES_PATH,
        coco_path=TRAIN_ANNOTATIONS,
        set_ratio=CONFIG.training.train_set_ratio,
        custom_categories=new_categories,
        batch_size=CONFIG.training.batch_size,
        shuffle=CONFIG.training.shuffle,
        transform_func=transforms if CONFIG.training.augmentation else None,
        desire_bbox_format=CONFIG.training.desire_bbox_format,
        pin_memory=CONFIG.training.pin_memory
    )
    valid_dataloader = None
    if getattr(CONFIG.paths.valid, 'images', None):
        if getattr(CONFIG.paths.valid, 'annotations', None):
            valid_dataloader = load_dataloader(
                root_dir=VALID_IMAGES_PATH,
                coco_path=VALID_ANNOTATIONS,
                set_ratio=CONFIG.training.valid_set_ratio,
                custom_categories=new_categories,
                batch_size=CONFIG.training.batch_size,
                shuffle=CONFIG.training.shuffle,
                transform_func=transforms if CONFIG.training.augmentation else None,
                desire_bbox_format=CONFIG.training.desire_bbox_format,
                pin_memory=CONFIG.training.pin_memory
            )
        else:
            raise ValueError('If you want to use validation dataset to evaluate model, you must also specify the path to the annotations.')

    # WandbLogger
    wandb_logger = None
    if CONFIG.training.log_metrics:
        wandb_logger = WandbLogger(
            project_name=CONFIG.wandb_logger.project_name, 
            project_config=OmegaConf.to_container(CONFIG.wandb_logger.model_config, resolve=True)  # Convert to dict
        )
    
    # Model
    deformable_detr_model = DefDetrModel(model_id=CONFIG.model.model_id, id2label=CONFIG.model.id2label, device=CONFIG.training.device)

    # Trainer
    trainer = TransformerTrainer(
        model=deformable_detr_model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        val_frequency=CONFIG.training.val_frequency,
        wandb_logger=wandb_logger,
        map_per_class=CONFIG.training.map_per_class,
        n_epochs=CONFIG.training.num_epochs,
        optimizer=CONFIG.optimizer.type,
        optimizer_params=CONFIG.optimizer.params,
        lr_scheduler=CONFIG.scheduler.type if getattr(CONFIG.scheduler, 'type', None) else None,
        lr_scheduler_params=CONFIG.scheduler.params if getattr(CONFIG.scheduler, 'type', None) else None
    )
    trainer.train()

if __name__ == '__main__':
    main()