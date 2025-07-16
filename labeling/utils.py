import os
import cv2
import numpy as np
from PySide6.QtGui import QImage
from config import Config

class Utils:
    @staticmethod
    def create_directory_structure(base_dir: str, project_name: str) -> tuple:
        """Creates project directory structure"""
        project_dir = os.path.join(base_dir, project_name)
        images_dir = os.path.join(project_dir, "images")
        labels_dir = os.path.join(project_dir, "labels")
        
        # Create directories if they don't exist
        for directory in [project_dir, images_dir, labels_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
        return project_dir, images_dir, labels_dir
    
    @staticmethod
    def resize_frame(frame):
        """Resize frame to standard size"""
        return cv2.resize(frame, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
    
    @staticmethod
    def create_empty_label_image():
        """Create an empty transparent image for labeling"""
        # Create a transparent image (RGBA with alpha=0)
        return np.zeros((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 4), dtype=np.uint8)
    
    @staticmethod
    def draw_line(image, pt1, pt2, thickness):
        """Draw a line on the image with semi-transparency"""
        # Create a color with alpha channel (Green with 50% opacity)
        color = (*Config.LINE_COLOR, 128)  # BGRA format with alpha=128 (50% opacity)
        cv2.line(image, pt1, pt2, color, thickness)
        return image
    
    @staticmethod
    def convert_cv_to_qimage(cv_image) -> QImage:
        """Converts OpenCV image to QImage for display"""
        height, width, channels = cv_image.shape
        bytes_per_line = channels * width
        
        if channels == 3:
            # Handle RGB images
            # Convert BGR to RGB
            cv_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            return QImage(cv_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        elif channels == 4:
            # Handle RGBA images
            # OpenCV uses BGRA, Qt uses RGBA
            cv_rgba = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGBA)
            return QImage(cv_rgba.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
        else:
            # Fallback for other formats
            return QImage()
