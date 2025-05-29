import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Define the dataset directory
DATASET_DIR = "ETIS-LaribPolypDB"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
MASKS_DIR = os.path.join(DATASET_DIR, "masks")

# Create directories for the processed dataset
PROCESSED_DIR = "preprocessed_data1"
TRAIN_IMAGES_DIR = os.path.join(PROCESSED_DIR, "train", "images")
TRAIN_MASKS_DIR = os.path.join(PROCESSED_DIR, "train", "masks")
VALID_IMAGES_DIR = os.path.join(PROCESSED_DIR, "valid", "images")
VALID_MASKS_DIR = os.path.join(PROCESSED_DIR, "valid", "masks")

os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
os.makedirs(TRAIN_MASKS_DIR, exist_ok=True)
os.makedirs(VALID_IMAGES_DIR, exist_ok=True)
os.makedirs(VALID_MASKS_DIR, exist_ok=True)

# Function to load and resize the image while preserving the aspect ratio
def load_image(image_path, target_size=(256, 256)):
    # Load the image
    image = cv2.imread(image_path)
    
    # Get the original dimensions
    h, w = image.shape[:2]
    
    # Compute the scaling factor to preserve the aspect ratio
    scale = min(target_size[0] / h, target_size[1] / w)
    
    # Calculate the new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h))
    
    # Create a canvas of the target size and place the resized image in the center
    padded_image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    x_offset = (target_size[1] - new_w) // 2
    y_offset = (target_size[0] - new_h) // 2
    padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    
    return padded_image

# Function to preprocess and binarize masks
def preprocess_mask(mask):
    # Convert mask to grayscale
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # Apply threshold to create binary mask
    _, binary_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask

# Function to create and write image-mask pairs for each file path in given directories
def create_and_write_image_mask(image_paths, mask_paths, save_images_dir, save_masks_dir):
    for image_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc="Processing images and masks"):
        image = load_image(image_path)
        mask = load_image(mask_path)
        
        # Preprocess the mask to convert it into a binary mask
        mask = preprocess_mask(mask)

        # Save the image and mask to the processed directories
        image_filename = os.path.basename(image_path)
        mask_filename = os.path.basename(mask_path)

        cv2.imwrite(os.path.join(save_images_dir, image_filename), image)
        cv2.imwrite(os.path.join(save_masks_dir, mask_filename), mask)

# Get a list of image and mask files in your dataset directories
image_files = sorted([os.path.join(IMAGES_DIR, filename) for filename in os.listdir(IMAGES_DIR)])
mask_files = sorted([os.path.join(MASKS_DIR, filename) for filename in os.listdir(MASKS_DIR)])

# Split the dataset into training and validation sets
train_images, valid_images, train_masks, valid_masks = train_test_split(image_files, mask_files, train_size=0.8, shuffle=True, random_state=42)

# Process and write the image-mask pairs for training and validation
create_and_write_image_mask(train_images, train_masks, TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR)
create_and_write_image_mask(valid_images, valid_masks, VALID_IMAGES_DIR, VALID_MASKS_DIR)
