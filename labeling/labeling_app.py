import sys
import os
import cv2
from sklearn.model_selection import train_test_split
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from config import Config

from ui import LabelingToolUI
from utils import Utils

class LabelingApp:
    def __init__(self):
        # Initialize app and UI
        self.app = QApplication(sys.argv)
        self.ui = LabelingToolUI()
        
        # Connect signals
        self.connect_signals()
        
        # Initialize variables
        self.video_capture = None
        self.frames = []
        self.label_images = []  # Store label images corresponding to frames
        self.head_index = -1
        self.project_dir = None
        self.images_dir = None
        self.labels_dir = None
        self.line_thickness = Config.DEFAULT_LINE_THICKNESS
        
        # Track line drawing history for each frame
        self.line_history = {}  # Dictionary: frame_index -> list of (pt1, pt2, thickness)
    
    def connect_signals(self):
        # Connect UI signals to methods
        self.ui.output_folder_selected.connect(self.on_output_folder_selected)
        self.ui.project_name_entered.connect(self.on_project_name_entered)
        self.ui.video_file_selected.connect(self.on_video_file_selected)
        self.ui.next_frame_requested.connect(self.show_next_frame)
        self.ui.previous_frame_requested.connect(self.show_previous_frame)
        self.ui.line_thickness_changed.connect(self.on_line_thickness_changed)
        self.ui.points_selected.connect(self.on_points_selected)
        self.ui.undo_requested.connect(self.on_undo_requested)
        self.ui.save_requested.connect(self.on_save_requested)
        self.ui.reset_requested.connect(self.on_reset_requested)
        self.ui.frame_selected.connect(self.on_frame_selected)
        self.ui.visualize_requested.connect(self.on_visualize_requested)
        self.ui.create_dataset_requested.connect(self.on_create_dataset_requested)
        self.ui.save_all_modified_requested.connect(self.on_save_all_modified_requested)
    
    def run(self):
        # Show UI and start event loop
        self.ui.show()
        
        # Start project setup workflow
        if not self.setup_project():
            return 1
        
        # Run the application
        return self.app.exec()
    
    def setup_project(self) -> bool:
        # Step 1: Select output folder
        if not self.ui.prompt_for_output_folder():
            self.ui.show_error("Error", "No output folder selected. Application will close.")
            return False
        
        # Step 2: Get project name
        if not self.ui.prompt_for_project_name():
            self.ui.show_error("Error", "No project name provided. Application will close.")
            return False
        
        return True
    
    def on_output_folder_selected(self, folder_path: str):
        # Store the selected output folder
        self.output_folder = folder_path
    
    def on_project_name_entered(self, project_name: str):
        # Create project structure with the provided name
        try:
            self.project_dir, self.images_dir, self.labels_dir = Utils.create_directory_structure(
                self.output_folder, project_name)
            
            self.ui.show_info("Success", 
                             f"Project created at: {self.project_dir}\n"
                             f"Images directory: {self.images_dir}\n"
                             f"Labels directory: {self.labels_dir}")
            
            # Enable video loading
            self.ui.enable_video_loading()
            
        except Exception as e:
            self.ui.show_error("Error", f"Failed to create project directories: {str(e)}")
            sys.exit()
    
    def on_video_file_selected(self, video_path: str):
        # Open the selected video file
        self.video_capture = cv2.VideoCapture(video_path)
        
        if not self.video_capture.isOpened():
            self.ui.show_error("Error", "Could not open video file.")
            return
        
        # Store frames and label images
        self.frames = []
        self.label_images = []
        self.head_index = -1
        self.line_thickness = Config.DEFAULT_LINE_THICKNESS
        
        # Track line drawing history for each frame
        self.line_history = {}  # Dictionary: frame_index -> list of (pt1, pt2, thickness)
        
        # Clear the frame list widget
        self.ui.clear_frame_list()
        
        # Update status
        video_name = os.path.basename(video_path)
        self.ui.update_status(f"Loaded video: {video_name}")
        
        # Enable navigation buttons
        self.ui.enable_navigation_buttons(False, True, True)
        
        # Show the first frame
        self.show_next_frame()
    
    def show_next_frame(self):

            
        # If we're at the end of loaded frames, try to get a new frame
        if not self.video_capture:
            return
            
        # Get the next frame
        ret, frame = self.video_capture.read()
        
        if not ret:
            # End of video reached
            self.video_capture.release()
            self.video_capture = None
            self.ui.update_status("End of video reached")
            return
        
        # Resize frame to display size
        frame = Utils.resize_frame(frame)
        
        # Add frame to the list
        self.frames.append(frame)
        
        # Create an empty label image
        label_image = Utils.create_empty_label_image()
        self.label_images.append(label_image)
        
        # Determine new frame index (last element)
        self.head_index = len(self.frames) - 1

        # Add frame to the frame list widget and select it
        self.ui.add_frame_to_list(self.head_index)
        self.ui.frame_list.setCurrentRow(self.head_index)
        
        # Display the frame and label
        self.display_current_frame()
        
        # Update navigation buttons state
        self.ui.enable_navigation_buttons(
            self.head_index > 0, 
            self.video_capture is not None, 
            True
        )
    
    def show_previous_frame(self):
        if self.head_index <= 0 or not self.frames:
            return
        
        # Move head index backward
        self.head_index -= 1
        
        # Display the frame and label
        self.display_current_frame()
        
        # Update navigation buttons state based on our new position
        self.ui.enable_navigation_buttons(
            self.head_index > 0,
            self.head_index < len(self.frames) - 1 or self.video_capture is not None,
            True
        )
    
    def display_current_frame(self):
        # Get current frame and label image
        frame = self.frames[self.head_index]
        label_image = self.label_images[self.head_index]
        
        # Convert frame to QImage and display
        frame_q_image = Utils.convert_cv_to_qimage(frame)
        frame_pixmap = QPixmap.fromImage(frame_q_image)
        self.ui.update_frame_display(frame_pixmap)
        
        # Convert label image to QImage and display
        mask_q_image = Utils.convert_cv_to_qimage(label_image)
        mask_pixmap = QPixmap.fromImage(mask_q_image)
        self.ui.update_mask_display(mask_pixmap)
        
        # Update undo button state based on whether there are lines to undo for this frame
        has_lines = self.head_index in self.line_history and len(self.line_history[self.head_index]) > 0
        self.ui.undo_button.setEnabled(has_lines)
        
    def on_line_thickness_changed(self, thickness):
        # Update line thickness
        self.line_thickness = thickness
        
    def on_undo_requested(self):
        # Undo the last line drawn on the current frame
        if self.head_index in self.line_history and self.line_history[self.head_index]:
            # Remove the last line from history
            self.line_history[self.head_index].pop()
            
            # Redraw the label image from scratch with the remaining lines
            # First, create a clean label image
            current_label = Utils.create_empty_label_image()
            
            # Redraw all the remaining lines
            for pt1, pt2, thickness in self.line_history[self.head_index]:
                Utils.draw_line(current_label, pt1, pt2, thickness)
            
            # Update the current label image
            self.label_images[self.head_index] = current_label
            
            # Update the mask display
            mask_q_image = Utils.convert_cv_to_qimage(current_label)
            mask_pixmap = QPixmap.fromImage(mask_q_image)
            self.ui.update_mask_display(mask_pixmap)
            
            # Disable undo button if no more lines to undo
            if not self.line_history[self.head_index]:
                self.ui.undo_button.setEnabled(False)
                
    def on_save_requested(self):
        # Save current frame and label if we have a valid frame
        if self.head_index >= 0 and self.head_index < len(self.frames) and self.project_dir:
            # Create the filename based on frame index
            filename = f"frame_{self.head_index:06d}"
            
            # Save the frame image
            frame_path = os.path.join(self.images_dir, f"{filename}.png")
            cv2.imwrite(frame_path, self.frames[self.head_index])
            
            # Convert the RGBA label to BGR format for saving
            label = self.label_images[self.head_index]
            # Remove alpha channel for saving (OpenCV prefers BGR)
            label_bgr = cv2.cvtColor(label, cv2.COLOR_BGRA2BGR)
            
            # Save the label image
            label_path = os.path.join(self.labels_dir, f"{filename}.png")
            cv2.imwrite(label_path, label_bgr)
            
            # Update status with minimal info (just the frame saved)
            self.ui.update_status(f"Saved frame_{self.head_index:06d}.png")
        else:
            # Show error if no frame or project directory
            if not self.project_dir:
                self.ui.show_error("Error", "Project directory not set. Please select an output folder and project name.")
            else:
                self.ui.show_error("Error", "No valid frame to save.")
                
    def on_reset_requested(self):
        # Reset (clear) all lines in the current label image
        if self.head_index >= 0 and self.head_index < len(self.label_images):
            # Create a new empty label image
            self.label_images[self.head_index] = Utils.create_empty_label_image()
            
            # Clear the line history for this frame
            if self.head_index in self.line_history:
                self.line_history[self.head_index] = []
            
            # Update the display
            mask_q_image = Utils.convert_cv_to_qimage(self.label_images[self.head_index])
            mask_pixmap = QPixmap.fromImage(mask_q_image)
            self.ui.update_mask_display(mask_pixmap)
            
            # Disable the undo button since we've cleared all lines
            self.ui.undo_button.setEnabled(False)
            
            # Update status
            self.ui.update_status(f"Reset label for frame_{self.head_index:06d}.png")
            
    def on_visualize_requested(self):
        """Blend images with their labels and save to a 'blended' folder inside the project."""
        if not self.project_dir:
            self.ui.show_error("Error", "Project directory not set. Please set up a project first.")
            return

        blended_dir = os.path.join(self.project_dir, "blended")
        if not os.path.exists(blended_dir):
            os.makedirs(blended_dir)

        image_files = [f for f in os.listdir(self.images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not image_files:
            self.ui.show_error("Error", "No images found in the images directory.")
            return

        blended_count = 0
        for img_name in image_files:
            image_path = os.path.join(self.images_dir, img_name)
            label_path = os.path.join(self.labels_dir, img_name)

            if not os.path.exists(label_path):
                continue

            image = cv2.imread(image_path)
            label = cv2.imread(label_path)
            if image is None or label is None:
                continue

            if image.shape[:2] != label.shape[:2]:
                label = cv2.resize(label, (image.shape[1], image.shape[0]))

            blended = cv2.addWeighted(image, 0.7, label, 0.3, 0)
            cv2.imwrite(os.path.join(blended_dir, img_name), blended)
            blended_count += 1

        if blended_count == 0:
            self.ui.show_error("Error", "No blended images were generated.")
        else:
            self.ui.show_info("Visualization Complete", f"Blended {blended_count} images saved to: {blended_dir}")
            self.ui.update_status(f"Blended {blended_count} images saved to 'blended' folder.")

    def on_create_dataset_requested(self):
        """Create PIDNet list files (train/val/test) from current project images/labels."""
        if not self.project_dir:
            self.ui.show_error("Error", "Project directory not set. Please set up a project first.")
            return

        lists_root = os.path.join(self.project_dir, "lists")
        os.makedirs(lists_root, exist_ok=True)

        img_files = {f for f in os.listdir(self.images_dir) if os.path.isfile(os.path.join(self.images_dir, f))}
        mask_files = {f for f in os.listdir(self.labels_dir) if os.path.isfile(os.path.join(self.labels_dir, f))}
        valid_files = sorted(img_files & mask_files)

        if not valid_files:
            self.ui.show_error("Error", "No matching image/label pairs found.")
            return

        train_files, test_files = train_test_split(valid_files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)

        def write_list(path: str, items: list[str]):
            with open(path, "w") as file:
                for name in items:
                    file.write(f"images/{name} labels/{name}\n")

        write_list(os.path.join(lists_root, "train.lst"), train_files)
        write_list(os.path.join(lists_root, "val.lst"), val_files)
        write_list(os.path.join(lists_root, "test.lst"), test_files)
        write_list(os.path.join(lists_root, "trainval.lst"), train_files + val_files)

        self.ui.show_info(
            "Dataset Lists Created",
            f"Lists saved to {lists_root}\nTrain: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}"
        )
        self.ui.update_status(
            f"Dataset lists generated (Train {len(train_files)}, Val {len(val_files)}, Test {len(test_files)})."
        )

    def on_save_all_modified_requested(self):
        """Save all frames that have drawn lines, skip frames without lines"""
        if not self.project_dir:
            self.ui.show_error("Error", "Project directory not set. Please set up a project first.")
            return

        if not self.frames:
            self.ui.show_error("Error", "No frames loaded.")
            return

        saved_count = 0
        skipped_count = 0

        for frame_index in range(len(self.frames)):
            # Check if this frame has any lines drawn
            has_lines = frame_index in self.line_history and len(self.line_history[frame_index]) > 0
            
            if has_lines:
                # Save this frame and its label
                filename = f"frame_{frame_index:06d}"
                
                # Save the frame image
                frame_path = os.path.join(self.images_dir, f"{filename}.png")
                cv2.imwrite(frame_path, self.frames[frame_index])
                
                # Convert the RGBA label to BGR format for saving
                label = self.label_images[frame_index]
                # Remove alpha channel for saving (OpenCV prefers BGR)
                label_bgr = cv2.cvtColor(label, cv2.COLOR_BGRA2BGR)
                
                # Save the label image
                label_path = os.path.join(self.labels_dir, f"{filename}.png")
                cv2.imwrite(label_path, label_bgr)
                
                saved_count += 1
            else:
                skipped_count += 1

        # Show summary message
        if saved_count > 0:
            self.ui.show_info("Save Complete", 
                            f"Saved {saved_count} modified frames.\n"
                            f"Skipped {skipped_count} frames without lines.")
            self.ui.update_status(f"Saved {saved_count} modified frames, skipped {skipped_count} empty frames.")
        else:
            self.ui.show_info("No Modified Frames", "No frames with drawn lines found to save.")
            self.ui.update_status("No modified frames found to save.")

    def on_frame_selected(self, frame_index):
        # Handle frame selection from the list widget
        if 0 <= frame_index < len(self.frames):
            # Update head index
            self.head_index = frame_index
            
            # Display the selected frame
            self.display_current_frame()
            
            # Update navigation buttons state
            self.ui.enable_navigation_buttons(
                self.head_index > 0,
                self.head_index < len(self.frames) - 1 or self.video_capture is not None,
                True
            )
        else:
            self.ui.show_error("Error", "Invalid frame index selected.")
        
    def on_points_selected(self, point1, point2):
        # Draw a line between the two points on the current label image
        if self.head_index >= 0 and self.head_index < len(self.label_images):
            # Convert QPoint to tuple for OpenCV
            pt1 = (point1.x(), point1.y())
            pt2 = (point2.x(), point2.y())
            
            # Draw the line on the label image
            current_label = self.label_images[self.head_index]
            Utils.draw_line(current_label, pt1, pt2, self.line_thickness)
            
            # Store this line in history for potential undo
            if self.head_index not in self.line_history:
                self.line_history[self.head_index] = []
            self.line_history[self.head_index].append((pt1, pt2, self.line_thickness))
            
            # Enable the undo button since we now have a line to undo
            self.ui.undo_button.setEnabled(True)
            
            # Update the mask display
            mask_q_image = Utils.convert_cv_to_qimage(current_label)
            mask_pixmap = QPixmap.fromImage(mask_q_image)
            self.ui.update_mask_display(mask_pixmap)
