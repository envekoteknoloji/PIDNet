#!/usr/bin/env python3

import os
import cv2
import numpy as np
import argparse
import random
from config import Config
from utils import Utils

class Augmentation:
    def __init__(self, project_dir: str = None, min_angle: int = -30, max_angle: int = 30, blur_kernel: tuple = (5, 5)):
        """Initialize augmentation with project directory and augmentation parameters"""
        self.project_dir = project_dir
        self.images_dir = None
        self.labels_dir = None
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.blur_kernel = blur_kernel
        
        if self.project_dir:
            self.setup_directories()
    
    def setup_directories(self):
        """Setup image and label directories based on project directory"""
        if not os.path.exists(self.project_dir):
            raise ValueError(f"Project directory does not exist: {self.project_dir}")
        
        self.images_dir = os.path.join(self.project_dir, "images")
        self.labels_dir = os.path.join(self.project_dir, "labels")
        
        if not os.path.exists(self.images_dir) or not os.path.exists(self.labels_dir):
            raise ValueError(f"Images or labels directory not found in project: {self.project_dir}")
    
    def set_project_dir(self, project_dir: str):
        """Set project directory and setup associated directories"""
        self.project_dir = project_dir
        self.setup_directories()
    
    def apply_blur(self, image, kernel_size=None):
        """Apply Gaussian blur to an image"""
        if kernel_size is None:
            kernel_size = self.blur_kernel
            
        # Only apply blur to RGB images, not label masks (which have alpha channel)
        if len(image.shape) > 2 and image.shape[2] == 4:
            # This is a label with alpha channel, don't blur it
            return image
        else:
            # Apply Gaussian blur
            return cv2.GaussianBlur(image, kernel_size, 0)
    
    def rotate_image(self, image, angle: float):
        """Rotate an image by the specified angle without distortion"""
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Calculate the diagonal length of the image as the safe size for rotation
        # This ensures we have enough space for the rotated image without cropping
        diagonal = int(np.sqrt(height**2 + width**2))
        
        # Create a square canvas with the diagonal length
        # This ensures no part of the image is cut off during rotation
        if len(image.shape) > 2 and image.shape[2] == 4:  # RGBA image
            square = np.zeros((diagonal, diagonal, 4), dtype=np.uint8)
        else:  # RGB image
            square = np.zeros((diagonal, diagonal, 3), dtype=np.uint8)
            
        # Calculate the position to paste the original image centered in the square
        x_offset = (diagonal - width) // 2
        y_offset = (diagonal - height) // 2
        
        # Paste the original image in the center of the square
        if len(image.shape) > 2 and image.shape[2] == 4:  # RGBA image
            square[y_offset:y_offset+height, x_offset:x_offset+width, :] = image
        else:  # RGB image
            square[y_offset:y_offset+height, x_offset:x_offset+width, :] = image
            
        # Rotate the square image around its center
        center = (diagonal // 2, diagonal // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_square = cv2.warpAffine(
            square, rotation_matrix, (diagonal, diagonal),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )
        
        # Crop the rotated image back to the original size, centered
        x_crop = (diagonal - width) // 2
        y_crop = (diagonal - height) // 2
        rotated = rotated_square[y_crop:y_crop+height, x_crop:x_crop+width]
        
        return rotated
    
    def augment_file(self, image_path: str, angle: float = None, apply_blur: bool = False):
        """Augment a single image and its label with rotation and optional blur"""
        # Generate random rotation angle if none provided
        if angle is None:
            angle = random.uniform(self.min_angle, self.max_angle)
        
        # Get the base filename without extension
        base_filename = os.path.basename(image_path)
        filename, ext = os.path.splitext(base_filename)
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return
        
        # Find corresponding label file
        label_path = os.path.join(self.labels_dir, f"{filename}{ext}")
        if not os.path.exists(label_path):
            print(f"Warning: No label file found for {image_path}")
            return
        
        # Load the label (with alpha channel)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if label is None:
            print(f"Warning: Could not read label {label_path}")
            return
        
        # Ensure label has alpha channel
        if label.shape[2] != 4:
            # Convert to RGBA if it's not already
            label_bgr = label if label.shape[2] == 3 else cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
            label = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2BGRA)
        
        # Apply rotation to both image and label
        rotated_image = self.rotate_image(image, angle)
        rotated_label = self.rotate_image(label, angle)
        
        # Apply blur only to the image if requested (not to the label)
        if apply_blur:
            rotated_image = self.apply_blur(rotated_image)
        
        # Create output filenames with appropriate suffix
        suffix = "_rot"
        if apply_blur:
            suffix += "_blur"
            
        augmented_image_path = os.path.join(
            self.images_dir, f"{filename}{suffix}{ext}"
        )
        augmented_label_path = os.path.join(
            self.labels_dir, f"{filename}{suffix}{ext}"
        )
        
        # Save augmented files
        cv2.imwrite(augmented_image_path, rotated_image)
        cv2.imwrite(augmented_label_path, rotated_label)
        
        return augmented_image_path, augmented_label_path
    
    def augment_all(self, apply_rotation=True, apply_blur=False):
        """Augment all images and labels in the project"""
        if not self.images_dir or not self.labels_dir:
            raise ValueError("Project directories not set up")
        
        # Get all files in images directory (exclude already augmented files)
        image_files = [f for f in os.listdir(self.images_dir) 
                      if os.path.isfile(os.path.join(self.images_dir, f)) 
                      and not "_rot" in f 
                      and not "_blur" in f]
        
        augmented_files = []
        for image_file in image_files:
            image_path = os.path.join(self.images_dir, image_file)
            
            # Determine what augmentations to apply
            if apply_rotation and apply_blur:
                # Apply both rotation and blur
                result = self.augment_file(image_path, apply_blur=True)
                if result:
                    augmented_files.append(result)
                    print(f"Applied rotation and blur: {image_file}")
            elif apply_rotation:
                # Apply only rotation
                result = self.augment_file(image_path, apply_blur=False)
                if result:
                    augmented_files.append(result)
                    print(f"Applied rotation: {image_file}")
            elif apply_blur:
                # Apply only blur (with zero rotation)
                result = self.augment_file(image_path, angle=0, apply_blur=True)
                if result:
                    augmented_files.append(result)
                    print(f"Applied blur: {image_file}")
        
        print(f"Augmented {len(augmented_files)} files")
        return augmented_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment images and labels with rotation and blur')
    parser.add_argument('--project-dir', type=str, required=True, help='Project directory containing images and labels folders')
    
    # Rotation options
    parser.add_argument('--no-rotation', action='store_true', help='Disable rotation augmentation')
    parser.add_argument('--min-angle', type=int, default=-30, help='Minimum rotation angle (default: -30)')
    parser.add_argument('--max-angle', type=int, default=30, help='Maximum rotation angle (default: 30)')
    
    # Blur options
    parser.add_argument('--blur', action='store_true', help='Enable Gaussian blur augmentation')
    parser.add_argument('--blur-kernel-size', type=int, default=5, help='Blur kernel size (default: 5)')
    
    args = parser.parse_args()
    
    # Create blur kernel tuple from size argument
    blur_kernel = (args.blur_kernel_size, args.blur_kernel_size)
    
    # Initialize the augmenter with all parameters
    augmenter = Augmentation(
        project_dir=args.project_dir,
        min_angle=args.min_angle,
        max_angle=args.max_angle,
        blur_kernel=blur_kernel
    )
    
    # Run augmentation with specified options
    augmenter.augment_all(
        apply_rotation=not args.no_rotation,
        apply_blur=args.blur
    )
