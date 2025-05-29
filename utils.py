import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MedicalDataset(Dataset):
    def __init__(self, *, image_paths, mask_paths, img_size, ds_mean, ds_std, is_train=False, apply_gaussian_blur=True):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths  
        self.is_train    = is_train
        self.img_size    = img_size
        self.ds_mean = ds_mean
        self.ds_std = ds_std
        self.apply_gaussian_blur = apply_gaussian_blur
        self.transforms  = self.setup_transforms(mean=self.ds_mean, std=self.ds_std)
 
    def __len__(self):
        return len(self.image_paths)
 
    def setup_transforms(self, *, mean, std):
        transforms = []
 
        # Augmentation 
        if self.is_train:
            transforms.extend([
                A.HorizontalFlip(p=0.5), 
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(scale_limit=0.12, rotate_limit=0.15, shift_limit=0.12, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.CLAHE(clip_limit=2.0, p=0.5),
                A.GridDistortion(p=0.5),
                A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=50, p=0.5),
                A.CoarseDropout(max_holes=8, max_height=self.img_size[1]//20, max_width=self.img_size[0]//20, min_holes=5, fill_value=0, mask_fill_value=0, p=0.5)
            ])
 
        # Preprocess transforms - Normalization and converting to PyTorch tensor format (HWC --> CHW).
        transforms.extend([
            A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensorV2(always_apply=True),  
        ])
        return A.Compose(transforms)
 
    def load_file(self, file_path, depth=0):
        file = cv2.imread(file_path, depth)
        if depth == cv2.IMREAD_COLOR:
            file = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)  
        resized_file = cv2.resize(file, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
        
        
        if depth == cv2.IMREAD_GRAYSCALE:
            _, resized_file = cv2.threshold(resized_file, 127, 255, cv2.THRESH_BINARY)
        
        return resized_file
 
    def __getitem__(self, index):
       
        image = self.load_file(self.image_paths[index], depth=cv2.IMREAD_COLOR)
        mask  = self.load_file(self.mask_paths[index],  depth=cv2.IMREAD_GRAYSCALE)

        
        transformed = self.transforms(image=image, mask=mask)
        image, mask = transformed["image"], transformed["mask"].to(torch.long)

        
        if self.apply_gaussian_blur:
            mask_np = mask.numpy().astype(np.uint8)
            mask_np = cv2.GaussianBlur(mask_np, (5, 5), 0)
            mask = torch.tensor(mask_np, dtype=torch.long)

        
        mask = (mask > 0).to(torch.long)

        return image, mask
