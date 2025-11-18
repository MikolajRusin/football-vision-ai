# FootballAI Project

## Overview
*FootballAI Project* is a computer vision project developed as part of a course in Computer Vision. The primary goal of this project is to track football players on the field using artificial intelligence. 
The system utilizes object detection models such as DETR to process video frames, annotate key elements of the game (such as players, referees, the ball, etc.).

This project has been refactored to enhance its modularity, scalability, and maintainability. The latest version incorporates improvements to both the underlying model architecture and the data processing pipeline.

## Project Background
This project was originally developed as part of a Computer Vision course, with the aim of applying machine learning techniques to real-world video analysis problems. 
Over time, the project has been refactored to improve its performance and extend its functionality, making it suitable for a variety of football game analysis tasks, including player tracking, event detection, and game summarization.

## Table of Contents
- [Overview](#overview)
- [Project Background](#project-background)
- [Project Structure](#project-structure)
- [Annotating Data](#annotating-data)
- [Loading Data](#loading-data)
- [Training Transformer Models](#training-transformer-models)

## Project Structure
```bash
football-ai/
├── configs/                       # Configuration files
│   └── config.yaml                  # Main configuration file
├── logger/                        # Logging utilities
│   └── wandb_logger.py              # Integration with WandB for logging
├── manager/                       # Checkpoint and model management
│   └── checkpoint_manager.py        # Checkpoint management functionality
├── models/                        # Model definitions and configurations
│   └── def_detr_model.py            # Model architecture for DETR
├── notebooks/                               # Jupyter Notebooks for experimentation
│   ├── annotate_data.ipynb                    # Notebook for data annotation
│   ├── def_detr_test_one_video.ipynb
│   ├── download_data.ipynb                    # Notebook for data download
│   ├── prepare_data_for_yolo_training.ipynb
│   ├── rt_detr_v2_test_one_video.ipynb
│   ├── test_load_dataset.ipynb                # Notebook for testing dataset loading
│   └── yolov11m_test_one_video.ipynb 
├── training/                      # Training scripts and configurations
│   ├── scripts/                     # Additional scripts for training
│   │   └── def_detr_train.py          # Main training script
│   └── trainer/                     # Trainer classes and utilities
│       ├── evaluator.py               # Model evaluation functionality
│       └── transformer_trainer.py     # Transformer model trainer
├── utils/                         # Utility functions and helpers
│   ├── data_utils/                  # Data handling utilities
│   │   ├── load_dataloader.py         # Dataloader management
│   │   └── load_dataset.py            # Dataset loading utilities
│   └── box_ops.py                   # Bounding box operations
├── .gitignore                     # Git ignore file
└── .venv/                         # Virtual environment for project dependencies
```

## Setting Environment Variables
Before downloading and training data, we need to set up the API keys that will be used in this project.

1. **ROBOFLOW_API_KEY**: This key is required because we will be downloading data and using models from the Roboflow website.
2. **WANDB_API_KEY**: This key is optional. You only need it if you want to track your training results with Weights & Biases (WandB). If you don’t plan to use WandB for logging or model tracking, you can skip this step.

To set up these variables, create a `.env` file in the project root and paste your API keys as follows:
```bash
ROBOFLOW_API_KEY=<your roboflow api key>
WANDB_API_KEY=<your wandb api key>
```
## Downloading Data
The dataset for this project has been obtained from two sources:

1. **Google Drive**: The data consists of short video clips from football matches, which were used for training and evaluation. To prevent almost identical frames from being selected, every 5th frame was chosen, with the frame selection controlled by setting `stride=5`. The data processing and splitting into folders is done in the notebook `notebooks/download_data.ipynb`.

2. **Roboflow**: The dataset provided by Roboflow already contained labeled frames, so there was no need to extract frames manually. The dataset was ready for training and evaluation without further preprocessing.

Both datasets are downloaded and processed in the notebook `notebooks/download_data.ipynb`. After running the notebook, a folder called `data` will be created in the project directory. It will contain the following structure with images extracted from the videos:
```bash
├── data/
│  ├── train/ 
│  │   └── images/ (905 images)
│  │       ├── image_1.jpg
│  │       ├── image_2.jpg
│  │       │       ...
│  │       └── image_xyz.jpg
│  └── valid/
│  │   └── images/ (124 images)
│  │       ├── image_1.jpg
│  │       ├── image_2.jpg
│  │       │       ...
│  │       └── image_xyz.jpg
│  ├── test/
│  │   └── images/  (93 images)
│  │       ├── image_1.jpg
│  │       ├── image_2.jpg
│  │       │       ...
│  │       └── image_xyz.jpg
│  │   
│  └── origin_videos (5 videos)
│
```
  
## Annotating Data
Once we have downloaded the data, we can proceed to annotate it. The notebook to annotate the data can be found in `notebooks/annotate_data.ipynb`.

To annotate the data, I used a pre-trained YOLO model, specifically trained for this type of image. This model was sourced from Roboflow. 
First, the model was tested on a few samples to verify its performance on the dataset. After confirming that the model performed well, the entire dataset was annotated with bounding boxes and labels.

The most important steps at this stage are:
1. The confidence threshold was set to 0.3 to avoid poor predictions and reduce clutter in the dataset.
2. The Intersection over Union (IoU) threshold was set to 0.5 to eliminate duplicate predictions and select the most confident prediction for each object.
3. The predictions were saved as the COCO JSON format, with bounding box coordinates (x, y, width, height) in absolute pixel values, where (x, y) refers to the top-left corner of the box.
> ⚠️ Note: Category identifiers saved at this stage start at 1, because several transformer-based models require index 0 to refer to the background. This is due to the architecture and operation of these models.

After running the notebook, each folder in the `data` directory will contain its own `coco_annotations` folder. Inside each `coco_annotations` folder, 
you will find an `annotations.json` file with the corresponding annotations for the images in that folder. The data folder should look like the following structure:
```bash
├── data/
│  ├── train/ 
│  │   └── images/
│  │   └── coco_annotations/
│  │       └── annotations.json
│  └── valid/
│  │   └── images/
│  │   └── coco_annotations/
│  │       └── annotations.json
│  ├── test/
│  │   └── images/
│  │   └── coco_annotations/
│  │       └── annotations.json
│  │   
│  └── origin_videos (5 videos)
│
```

## Loading Data

The project uses a custom dataset loader and a wrapper function for building PyTorch dataloaders.  
These utilities ensure that images, annotations, and augmentations are handled in a consistent and model-compatible way.

---

### `LoadDataset`

`LoadDataset` is a custom class that inherits from PyTorch’s `Dataset`.  
Its purpose is to:

- load images from disk,
- parse COCO-style annotations,
- apply Albumentations transforms (if provided),
- convert bounding boxes to the desired output format,
- return data in the `(image, target)` structure required by transformer-based detection models.

You can find the implementation in: `utils/data_utils/load_dataset.py`

#### Arguments
| Argument | Type | Description |
|---------|------|-------------|
| `dataset_dir_path` | `Path` | Path to the directory containing image files (`train/images`, `valid/images`, etc.). |
| `coco_annotations_path` | `Path` | Path to the COCO `annotations.json` file for the dataset split. |
| `set_ratio` | `float \| int \| None` | Allows loading only a selected percentage or number of samples. Useful for debugging. |
| `transforms` | `albumentations.Compose \| None` | Image augmentations applied during loading. |
| `desire_bbox_format` | `str` | The required bounding box format (`xywh`, `cxcywh`, etc.). |
| `return_img_path` | `bool` | If `True`, the dataset also returns the image path together with the sample. |

---

### `load_dataloader`

The `load_dataloader` function is a convenient wrapper around `LoadDataset` and PyTorch’s `DataLoader`.  
It builds a fully configured dataloader for training or validation.

You can find it in: `utils/data_utils/load_dataloader.py`

#### Functionality

- Initializes a `LoadDataset` instance,
- optionally loads a custom category mapping,
- wraps the dataset in a PyTorch `DataLoader`,
- applies a custom `collate_fn` to support variable numbers of objects per image.

#### Arguments

| Argument | Type | Description |
|---------|------|-------------|
| `root_dir` | `Path \| str` | Path to the folder with image files. |
| `coco_path` | `Path \| str` | Path to the COCO annotation JSON file. |
| `set_ratio` | `float \| int \| None` | Same as in `LoadDataset`: limits the dataset size. |
| `custom_categories` | `dict[str, Any] \| None` | Optional custom class-name–to-id mapping. |
| `batch_size` | `int` | Number of samples per batch. |
| `shuffle` | `bool` | Whether to shuffle the dataset. |
| `transform_func` | `albumentations.Compose \| None` | Albumentations pipeline applied inside the dataset. |
| `desire_bbox_format` | `str` | Bounding box output format expected by the model. |
| `pin_memory` | `bool` | Enables `pin_memory=True` for faster host-to-GPU transfers. |

## Training Transformer Models
To train transformer-based object detection models (e.g., Deformable DETR or RT-DETR-V2), a custom training class named `TransformerTrainer` was implemented.  
Although the Hugging Face `Trainer` class could be used, a custom solution was chosen intentionally — mainly to maintain full control over the training loop, support additional features (e.g., mAP per class, frequency-based validation, custom checkpointing), and to better understand the mechanics behind training transformer architectures.

The full implementation of the trainer can be found in: `training/trainer/transformer_trainer.py`

---

### `TransformerTrainer`

`TransformerTrainer` is a flexible training module designed specifically for object detection models built with the Hugging Face Transformers library.  
It handles the following:

- full training loop with epoch- and iteration-level logging,
- computation and aggregation of model losses (`loss`, `loss_bbox`),
- configurable optimizer and learning-rate scheduler,
- validation at custom frequency,
- mAP evaluation (via the custom `Evaluator` class),
- checkpoint saving using `ModelCheckpointManager`,
- logging to Weights & Biases using `WandbLogger`.

The trainer accepts a wide range of parameters, allowing the user to precisely configure the training process.

#### Key Arguments

| Argument | Type | Description |
|---------|------|-------------|
| `model` | `nn.Module` | The transformer detection model to train. |
| `train_dataloader` | `DataLoader` | Dataloader for the training split. |
| `valid_dataloader` | `DataLoader \| None` | Optional validation dataloader. |
| `frequency_validating` | `int \| None` | Allows running validation every N iterations. |
| `score_threshold` | `float` | Score threshold for filtering model predictions during evaluation. |
| `n_epochs` | `int` | Number of training epochs. |
| `optimizer` | `str` | One of: `"adamw"`, `"adam"`, `"sgd"`. |
| `optimizer_params` | `dict \| None` | Parameters passed to the optimizer (supports custom backbone LR). |
| `lr_scheduler` | `str \| None` | One of: `"cosine_annealing"`, `"step"`, `"onecycle"`. |
| `lr_scheduler_params` | `dict \| None` | Parameters for the chosen scheduler. |
| `checkpoint_manager` | `ModelCheckpointManager \| None` | Utility for saving model checkpoints. |
| `frequency_saving_checkpoint` | `int \| None` | Allows saving checkpoints every N iterations. |
| `wandb_logger` | `WandbLogger \| None` | Optional Weights & Biases logger. |
| `map_per_class` | `bool` | Enables mAP calculation for each class separately. |

---

### Training Loop Overview

The trainer follows a classical loop structure:

1. **Forward pass**  
   The model receives a batch and returns all loss components through `outputs.loss` and `outputs.loss_dict`.

2. **Backward pass & optimizer step**  
   Loss gradients are cleared, computed, and applied.

3. **Logging**  
   - iteration-level logging (optional, to W&B),  
   - epoch summaries (loss, bbox loss, mAP).

4. **Validation**  
   - can run once per epoch or based on `frequency_validating`,  
   - predictions and targets are post-processed for the `Evaluator`.

5. **Checkpointing**  
   - at the end of each epoch,  
   - optionally every *N* iterations.

This setup provides full transparency and control over training — which was one of the key reasons to implement a custom trainer instead of relying on the built-in Hugging Face `Trainer`.

---

### `Training DefDetrModel`

`DefDetrModel` is a wrapper module built around the Hugging Face  
`DeformableDetrForObjectDetection` architecture.  
The goal of this class is to provide a clean, unified interface for:

- loading pretrained Deformable DETR models,
- customizing label mappings (`id2label`),
- resetting classification heads for fine-tuning on new datasets,
- performing forward passes for both training and inference,
- loading checkpoints saved during training,
- loading models directly from a Hugging Face Hub repository.

This wrapper integrates seamlessly with the `TransformerTrainer` and ensures that  
images, annotations, and predictions are processed in a format suitable for  
transformer-based object detection.

---

#### Key Features

##### ✔️ Configurable label space  
If `id2label` is passed during initialization, the model automatically:

- updates `config.id2label`,
- sets `config.label2id`,
- adjusts the classification head for the correct number of classes.

This makes the model fully adaptable to custom datasets (e.g., your FootballAI dataset).

##### ✔️ Optional head reset  
When `reset_head=True`, the classification layers are re-initialized using Xavier initialization —  
useful when fine-tuning on newly annotated datasets with different label distributions.

##### ✔️ Built-in processor  
The model loads `DeformableDetrImageProcessor` and uses it internally for:

- image normalization,
- padding,
- conversion of annotations to model-friendly tensors.

This allows you to pass raw images and COCO-like annotations directly to the model.

##### ✔️ Seamless checkpoint loading  
The `load_model_checkpoint()` method loads:

- full model weights,
- full config,
- restores label mappings,
- reloads/reinitializes heads if the number of classes has changed.

Compatible with checkpoints saved by your `ModelCheckpointManager`.

##### ✔️ Hugging Face Hub support  
The `from_pretrained()` class method allows loading the model directly from a Hub repository  
including `.safetensors` weights.  
This is especially useful for sharing trained detectors with collaborators.

---




