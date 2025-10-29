from pathlib import Path
import sys
import os
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.data.load_dataloader import load_dataloader
from models.def_detr_model import DefDetrModel
import albumentations as A
import argparse
import torch

# Path of the project
PROJECT_PATH = Path(__file__).resolve().parent.parent
# Main data dir
DATA_PATH = PROJECT_PATH / 'data'
# Paths to train, valid and test dir
TRAIN_PATH = DATA_PATH / 'train'
TRAIN_IMAGES_PATH = TRAIN_PATH / 'images'
TRAIN_ANNOTATIONS = TRAIN_PATH / 'coco_annotations' / 'annotations.json'
VALID_PATH = DATA_PATH / 'valid'
VALID_IMAGES_PATH = VALID_PATH / 'images'
VALID_ANNOTATIONS = VALID_PATH / 'coco_annotations' / 'annotations.json'
TEST_PATH = DATA_PATH / 'test'
TEST_IMAGES_PATH = TEST_PATH / 'images'
TEST_ANNOTATIONS = TEST_PATH / 'coco_annotations' / 'annotations.json'

MODEL_ID = 'SenseTime/deformable-detr'
ID2LABEL = {
    0: 'N/A',
    1: 'Ball', 
    2: 'Goalkeeper', 
    3: 'Player', 
    4: 'Referee'
}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs. Defaults to 10.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size. Defaults to 2.')
    parser.add_argument('--train_set_ratio', type=float, default=1.0, help='Ratio of the datasets (0.0 - 1.0] or an integer for number of images. Defaults to None.')
    parser.add_argument('--valid_set_ratio', type=float, default=None, help='Ratio of the datasets (0.0 - 1.0] or an integer for number of images. Defaults to None.')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle dataset')
    parser.add_argument('--normalize_boxes', action='store_true', help='Normalize boxes before passing them into the model')
    parser.add_argument('--augmentation', action='store_true', help='Apply data augmentation')
    parser.add_argument('--pin_memory', action='store_true', help='Apply pin memory')
    parser.add_argument('--device', type=str, default='cpu', help='Device where the model will be stored')
    # parser.add_argument('--backbone_lr', type=float, default=None, help="Learning rate for the backbone of the model. If None, the model will use the default learning rate ('lr') for all parameters. Defaults to None.")
    # parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate for all model parameters, or for the non-backbone layers if 'backbone_lr' is set. Defaults to 0.0001.")
    # parser.add_argument('--val_frequency', type=float, default=None, help='Frequency of model validation during epochs.')
    # parser.add_argument('--save_checkpoints', type=bool, default=False, help='If True then the model save checkpoints.')
    # parser.add_argument('--max_checkpoints', type=int, default=10, help='Maximum number of checkpoints contained in a folder.')
    # parser.add_argument('--use_scheduler', type=bool, default=False, help='If True then use learning rate scheduler (CosineAnnealingLR).')
    # parser.add_argument('--T_max', type=int, default=5, help='Number of epochs to reach minimum learning rate -> if use_scheduler=True.')
    # parser.add_argument('--eta_min', type=float, default=1e-6, help='Minimum learning rate after T_max epochs -> if use_scheduler=True.')
    # parser.add_argument('--log_results', type=bool, default=False, help='If True then log results in Weights & Biases (wandb)')
    # parser.add_argument('--project_name', type=str, default='Model_fine_tuning', help='Name of the created project in Weights & Biases (wandb)')
    args = parser.parse_args()

    # === Log used parameters ===
    print('\n' + '=' * 80)
    print('Training Configuration')
    print('=' * 80)
    for arg, value in vars(args).items():
        print(f'{arg:20s}: {value}')
    print('=' * 80 + '\n')

    return args

def main():
    args = parse_arguments()

    # Augmentation function
    transforms = A.Compose(
        [
            A.HorizontalFlip(p=0.4),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomRotate90(p=0.3),  # Rotacja obrazu o 90 stopni
            A.CLAHE(p=0.5),  # Stosowanie CLAHE
            A.HueSaturationValue(p=0.3)  # Zmiana odcienia, nasycenia i wartości
        ],
        bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'])
    )

    # Train and Valid dataloaders
    new_categories = [
        {'id': k, 'name': v}
        for k, v in ID2LABEL.items()
    ]
    train_dataloader = load_dataloader(
        root_dir=TRAIN_IMAGES_PATH,
        coco_path=TRAIN_ANNOTATIONS,
        set_ratio=args.train_set_ratio,
        custom_categories=new_categories,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        transform_func=transforms if args.augmentation else None,
        normalize_boxes=args.normalize_boxes,
        pin_memory=args.pin_memory
    )
    valid_dataloader = None
    if args['valid_set_ratio'] is not None:
        valid_dataloader = load_dataloader(
            root_dir=TRAIN_IMAGES_PATH,
            coco_path=TRAIN_ANNOTATIONS,
            set_ratio=args.valid_set_ratio,
            custom_categories=new_categories,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            transform_func=transforms if args.augmentation else None,
            normalize_boxes=args.normalize_boxes,
            pin_memory=args.pin_memory
        )

    # Model
    device = args['device']
    deformable_detr_model = DefDetrModel(model_id=MODEL_ID, id2label=ID2LABEL, device=device)

if __name__ == '__main__':
    main()