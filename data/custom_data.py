import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from albumentations import (
    Compose, Resize, Normalize, HorizontalFlip, VerticalFlip, 
    RandomBrightnessContrast, GridDistortion, ElasticTransform, 
    RandomCrop, CLAHE, ShiftScaleRotate, CoarseDropout
)
from albumentations.pytorch import ToTensorV2

class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, patches=False, patch_size=(512, 768)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.patches = patches
        self.patch_size = patch_size

    def __len__(self):
        return len(self.image_paths)

    def tiles(self, img, mask):
        img_patches = img.unfold(1, self.patch_size[0], self.patch_size[0]).unfold(2, self.patch_size[1], self.patch_size[1])
        img_patches = img_patches.contiguous().view(3, -1, self.patch_size[0], self.patch_size[1])
        img_patches = img_patches.permute(1, 0, 2, 3)

        mask_patches = mask.unfold(0, self.patch_size[0], self.patch_size[0]).unfold(1, self.patch_size[1], self.patch_size[1])
        mask_patches = mask_patches.contiguous().view(-1, self.patch_size[0], self.patch_size[1])

        return img_patches, mask_patches

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)  

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if self.patches:
            image, mask = self.tiles(image, mask)

        return image, mask


# Data transformations for training with augmentations
train_transform = Compose([
    RandomCrop(height=512, width=768, p=0.5),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomBrightnessContrast(p=0.5),
    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.5),
    GridDistortion(p=0.5),
    ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
    CLAHE(clip_limit=2.0, p=0.5),
    CoarseDropout(max_holes=8, max_height=64, max_width=64, min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


valid_transform = Compose([
    Resize(512, 768),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Preprocessed data path
data_dir = "preprocessed_data1"

# paths to image and mask directories
train_image_dir = os.path.join(data_dir, "train", "images")
train_mask_dir = os.path.join(data_dir, "train", "masks")

train_image_paths = [os.path.join(train_image_dir, filename) for filename in os.listdir(train_image_dir)]
train_mask_paths = [os.path.join(train_mask_dir, filename) for filename in os.listdir(train_mask_dir)]

train_dataset = CustomDataset(train_image_paths, train_mask_paths, transform=train_transform, patches=True)

valid_image_dir = os.path.join(data_dir, "valid", "images")
valid_mask_dir = os.path.join(data_dir, "valid", "masks")

valid_image_paths = [os.path.join(valid_image_dir, filename) for filename in os.listdir(valid_image_dir)]
valid_mask_paths = [os.path.join(valid_mask_dir, filename) for filename in os.listdir(valid_mask_dir)]

valid_dataset = CustomDataset(valid_image_paths, valid_mask_paths, transform=valid_transform, patches=False)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=10, shuffle=False)
