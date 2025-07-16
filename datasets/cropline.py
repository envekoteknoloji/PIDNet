# ------------------------------------------------------------------------------
# CropLine dataset for sugarbeet cropline segmentation
# ------------------------------------------------------------------------------

import os
import numpy as np
import random
from PIL import Image
import cv2

from .base_dataset import BaseDataset

class CropLine(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_classes=2,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=640,
                 crop_size=(480, 640),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(CropLine, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std)

        self.class_weights = None

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.flip = flip
        self.bd_dilate_size = bd_dilate_size
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]
        self.files = self.read_files()
        
        # Define colors for visualization: black (background) and green (cropline)
        self.color_list = [[0, 0, 0], [0, 255, 0]]
    
    def read_files(self):
        files = []
        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name
            })
        return files
        
    def color2label(self, color_map):
        label = np.ones(color_map.shape[:2]) * self.ignore_label
        for i, v in enumerate(self.color_list):
            label[(color_map == v).sum(2) == 3] = i
        return label.astype(np.uint8)
    
    def label2color(self, label):
        color_map = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
        for i, v in enumerate(self.color_list):
            color_map[label == i] = self.color_list[i]
        return color_map.astype(np.uint8)
    
    def random_rotate(self, image, label, max_angle=30):
        """Randomly rotate image and label within a range of angles.
        
        Args:
            image: Image array (H, W, 3)
            label: Label array (H, W)
            max_angle: Maximum rotation angle in degrees
        
        Returns:
            Rotated image and label
        """
        # Determine if we should rotate (50% chance)
        if random.random() < 0.5:
            return image, label
            
        # Generate a random angle between -max_angle and max_angle
        angle = random.uniform(-max_angle, max_angle)
        
        # Get the image dimensions
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation to image
        rotated_image = cv2.warpAffine(
            image, 
            rotation_matrix, 
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        
        # Apply rotation to label - use INTER_NEAREST to prevent interpolation between class values
        rotated_label = cv2.warpAffine(
            label, 
            rotation_matrix, 
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0))  # Use 0 (background) instead of ignore_label for border
        
        # Fix any label pixels that may have become ignore_label
        rotated_label[rotated_label == self.ignore_label] = 0
        
        return rotated_image, rotated_label
        
    def random_blur(self, image, label, prob=0.5, kernel_range=(3, 7)):
        """Randomly apply Gaussian blur to the image.
        
        Args:
            image: Image array (H, W, 3)
            label: Label array (H, W)
            prob: Probability of applying blur
            kernel_range: Range of kernel sizes for Gaussian blur
        
        Returns:
            Blurred image and unchanged label
        """
        # Determine if we should blur
        if random.random() > prob:
            return image, label
            
        # Generate a random kernel size (must be odd)
        kernel_size = random.randrange(kernel_range[0], kernel_range[1], 2)
        
        # Apply Gaussian blur to the image
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Label is not affected by blur
        return blurred_image, label
        
    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        
        # Load image
        image = Image.open(os.path.join(self.root, 'cropline', item["img"])).convert('RGB')
        image = np.array(image)
        size = image.shape

        # Load label
        label_img = Image.open(os.path.join(self.root, 'cropline', item["label"])).convert('RGB')
        color_map = np.array(label_img)
        label = self.color2label(color_map)
        
        # Apply rotation augmentation
        #image, label = self.random_rotate(image, label)
        
        # Apply blur augmentation
        #image, label = self.random_blur(image, label)

        # Use default augmentation from base dataset
        image, label, edge = self.gen_sample(image, label, 
                                           self.multi_scale, self.flip, 
                                           edge_pad=False, edge_size=self.bd_dilate_size, 
                                           city=False)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.label2color(preds[i])
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))