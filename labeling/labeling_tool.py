#!/usr/bin/env python3

import sys
import os
import cv2
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                              QHBoxLayout, QWidget, QFileDialog, QGraphicsView, 
                              QGraphicsScene, QLabel, QInputDialog, QMessageBox)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

from config import Config
from utils import Utils

class ImageLabelingTool(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize variables
        self.video_capture = None
        self.frames = []
        self.head_index = -1
        self.project_dir = None
        self.images_dir = None
        self.labels_dir = None
        
        # Setup UI
        self.setWindowTitle(Config.WINDOW_TITLE)
        self.resize(Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT)
        self.setup_ui()
        
        # Start project setup workflow
        self.setup_project()
    
    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Create graphics view for displaying images
        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.graphics_view)
        
        # Status label
        self.status_label = QLabel("No video loaded")
        main_layout.addWidget(self.status_label)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        # Previous button
        self.prev_button = QPushButton("Previous")
        self.prev_button.setFixedSize(Config.BUTTON_WIDTH, Config.BUTTON_HEIGHT)
        self.prev_button.clicked.connect(self.show_previous_frame)
        self.prev_button.setEnabled(False)
        button_layout.addWidget(self.prev_button)
        
        # Next button
        self.next_button = QPushButton("Next")
        self.next_button.setFixedSize(Config.BUTTON_WIDTH, Config.BUTTON_HEIGHT)
        self.next_button.clicked.connect(self.show_next_frame)
        self.next_button.setEnabled(False)
        button_layout.addWidget(self.next_button)
        
        # Load video button
        self.load_video_button = QPushButton("Load Video")
        self.load_video_button.setFixedSize(Config.BUTTON_WIDTH, Config.BUTTON_HEIGHT)
        self.load_video_button.clicked.connect(self.load_video)
        self.load_video_button.setEnabled(False)
        button_layout.addWidget(self.load_video_button)
        
        main_layout.addLayout(button_layout)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def setup_project(self):
        """Initial project setup workflow"""
        # Step 1: Select output folder
        output_folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not output_folder:
            QMessageBox.critical(self, "Error", "No output folder selected. Application will close.")
            sys.exit()
        
        # Step 2: Get project name
        project_name, ok = QInputDialog.getText(self, "Project Name", "Enter project name:")
        if not ok or not project_name:
            QMessageBox.critical(self, "Error", "No project name provided. Application will close.")
            sys.exit()
        
        # Step 3-4: Create directory structure
        try:
            self.project_dir, self.images_dir, self.labels_dir = Utils.create_directory_structure(
                output_folder, project_name)
            QMessageBox.information(self, "Success", 
                                   f"Project created at: {self.project_dir}\n"
                                   f"Images directory: {self.images_dir}\n"
                                   f"Labels directory: {self.labels_dir}")
            
            # Enable loading video now that we have set up the project
            self.load_video_button.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create project directories: {str(e)}")
            sys.exit()
    
    def load_video(self):
        """Load a video file"""
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        
        if not video_path:
            return
        
        # Open the video file
        self.video_capture = cv2.VideoCapture(video_path)
        if not self.video_capture.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video file.")
            return
        
        # Reset frames and head index
        self.frames = []
        self.head_index = -1
        
        # Update status
        video_name = os.path.basename(video_path)
        self.status_label.setText(f"Loaded video: {video_name}")
        
        # Enable navigation buttons
        self.next_button.setEnabled(True)
        self.show_next_frame()
    
    def show_next_frame(self):
        """Show the next frame from the video"""
        if self.video_capture is None:
            return
        
        # Check if we need to read a new frame or use one from memory
        if self.head_index >= len(self.frames) - 1:
            # Read a new frame from video
            ret, frame = self.video_capture.read()
            if not ret:
                QMessageBox.information(self, "End of Video", "Reached the end of the video.")
                return
            
            # Add frame to our list
            self.frames.append(frame)
            
            # Check if we need to remove old frames to save memory
            if len(self.frames) > Config.MAX_FRAMES_IN_MEMORY:
                # Remove the oldest frame (assuming we won't go back that far)
                self.frames.pop(0)
                # Adjust head_index since we removed a frame
                self.head_index = max(0, self.head_index - 1)
        
        # Move head index forward
        self.head_index += 1
        
        # Display the frame
        self.display_frame(self.frames[self.head_index])
        
        # Update status
        self.status_label.setText(f"Frame: {self.head_index + 1} / {len(self.frames)}")
        
        # Enable/disable navigation buttons
        self.prev_button.setEnabled(self.head_index > 0)
    
    def show_previous_frame(self):
        """Show the previous frame"""
        if self.head_index <= 0 or not self.frames:
            return
        
        # Move head index backward
        self.head_index -= 1
        
        # Display the frame
        self.display_frame(self.frames[self.head_index])
        
        # Update status
        self.status_label.setText(f"Frame: {self.head_index + 1} / {len(self.frames)}")
        
        # Enable/disable navigation buttons
        self.prev_button.setEnabled(self.head_index > 0)
    
    def display_frame(self, frame):
        """Display an OpenCV frame on the QGraphicsView"""
        # Convert frame to QImage
        q_image = Utils.convert_cv_to_qimage(frame)
        
        # Convert to QPixmap and display
        pixmap = QPixmap.fromImage(q_image)
        
        # Clear previous items and add the new pixmap
        self.graphics_scene.clear()
        self.graphics_scene.addPixmap(pixmap)
        
        # Adjust view
        self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.KeepAspectRatio)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageLabelingTool()
    window.show()
    sys.exit(app.exec())
