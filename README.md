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
  
## Downloading Data

The dataset for this project has been obtained from two sources:

1. **Google Drive**: The data consists of short video clips from football matches, which were used for training and evaluation. To prevent almost identical frames from being selected, every 5th frame was chosen, with the frame selection controlled by setting `stride=5`. The data processing and splitting into folders is done in the notebook `notebooks/download_data.ipynb`.

2. **Roboflow**: The dataset provided by Roboflow already contained labeled frames, so there was no need to extract frames manually. The dataset was ready for training and evaluation without further preprocessing.

Both datasets are downloaded and processed in the notebook `notebooks/download_data.ipynb`. After running the notebook, a folder called `data` will be created in the project directory. It will contain the following structure with images extracted from the videos:
```bash
├── data/
│  ├── train/ 
│  │   └── images (905 images)
│  └── valid/
│  │   └── images (124 images)
│  ├── test/
│  │   └── images  (93 images)
│  │   
│  └── origin_videos (5 videos)
│
```
