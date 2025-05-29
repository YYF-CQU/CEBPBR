import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import zipfile
import platform
import warnings
from glob import glob
from dataclasses import dataclass
 
warnings.filterwarnings("ignore", category=UserWarning)
 
import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt
 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
 
 
# For data augmentation and preprocessing.
import albumentations as A
from albumentations.pytorch import ToTensorV2
 

from transformers import SegformerForSemanticSegmentation
 
import lightning.pytorch as pl
from pytorch_lightning.loggers import TensorBoardLogger

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassF1Score
from torch.utils.tensorboard import SummaryWriter
 

from torchinfo import summary
 
torch.set_float32_matmul_precision('high')

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

%matplotlib inline


from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES: int = 1  
    IMAGE_SIZE: Tuple[int, int] = (192, 192)  # W, H
    MEAN: tuple = (0.485, 0.456, 0.406)
    STD:  tuple = (0.229, 0.224, 0.225)
    CHANNELS: int = 1  # single-channel grayscale
    BACKGROUND_CLS_ID: int = 0
    URL: str = r"https://www.kaggle.com/datasets"
    DATASET_PATH: str = os.path.join(os.getcwd(), "preprocessed_data1")
 
@dataclass(frozen=True)
class Paths:
    DATA_TRAIN_IMAGES: str = os.path.join(DatasetConfig.DATASET_PATH, "train", "images", r"*.png")
    DATA_TRAIN_LABELS: str = os.path.join(DatasetConfig.DATASET_PATH, "train", "masks",  r"*.png")
    DATA_VALID_IMAGES: str = os.path.join(DatasetConfig.DATASET_PATH, "valid", "images", r"*.png")
    DATA_VALID_LABELS: str = os.path.join(DatasetConfig.DATASET_PATH, "valid", "masks",  r"*.png")
         
@dataclass
class TrainingConfig:
    BATCH_SIZE:      int = 4 
    NUM_EPOCHS:      int = 50
    INIT_LR:       float = 3e-4
    NUM_WORKERS:     int = 0 if platform.system() == "Windows" else 12 # os.cpu_count()
 
    OPTIMIZER_NAME:  str = "AdamW"
    WEIGHT_DECAY:  float = 1e-4
    USE_SCHEDULER:  bool = True 
    SCHEDULER:       str = "MultiStepLR" 
    MODEL_NAME:      str = "CEBPBR"
    
     
 
@dataclass
class InferenceConfig:
    BATCH_SIZE:  int = 4
    NUM_BATCHES: int = 1
    
    
    


