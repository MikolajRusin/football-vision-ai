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

## Project Structure
```bash
football-ai/
├── configs/                      # Configuration files
│   └── config.yaml               # Main configuration file
├── logger/                       # Logging utilities
│   └── wandb_logger.py           # Integration with WandB for logging
├── manager/                      # Checkpoint and model management
│   └── checkpoint_manager.py     # Checkpoint management functionality
├── models/                       # Model definitions and configurations
│   └── def_detr_model.py         # Model architecture for DETR
├── notebooks/                    # Jupyter Notebooks for experimentation
│   ├── annotate_data.ipynb       # Notebook for data annotation
│   ├── download_data.ipynb       # Notebook for data download
│   └── test_load_dataset.ipynb   # Notebook for testing dataset loading
├── training/                     # Training scripts and configurations
│   ├── scripts/                  # Additional scripts for training
│   │   └── def_detr_train.py     # Main training script
│   └── trainer/                  # Trainer classes and utilities
│       ├── evaluator.py          # Model evaluation functionality
│       └── transformer_trainer.py# Transformer model trainer
├── utils/                        # Utility functions and helpers
│   ├── data_utils/               # Data handling utilities
│   │   ├── load_dataloader.py    # Dataloader management
│   │   └── load_dataset.py       # Dataset loading utilities
│   └── box_ops.py                # Bounding box operations
├── .gitignore                    # Git ignore file
└── .venv/                        # Virtual environment for project dependencies
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
