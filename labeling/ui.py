from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                              QWidget, QPushButton, QFileDialog, QLabel,
                              QLineEdit, QGraphicsView, QGraphicsScene, QSpinBox,
                              QMessageBox, QGraphicsPixmapItem, QGraphicsLineItem,
                              QListWidget, QGroupBox, QInputDialog, QTextEdit)
from PySide6.QtGui import QPixmap, QPen, QColor
from PySide6.QtCore import Qt, Signal, QPoint

from config import Config

class LabelingToolUI(QMainWindow):
    # Define signals for communication with the main application
    output_folder_selected = Signal(str)
    project_name_entered = Signal(str)
    video_file_selected = Signal(str)
    next_frame_requested = Signal()
    previous_frame_requested = Signal()
    line_thickness_changed = Signal(int)
    points_selected = Signal(QPoint, QPoint)
    undo_requested = Signal()
    save_requested = Signal()
    reset_requested = Signal()
    frame_selected = Signal(int)
    visualize_requested = Signal()
    create_dataset_requested = Signal()
    
    def __init__(self):
        super().__init__()
        
        # Setup UI
        self.setWindowTitle(Config.WINDOW_TITLE)
        self.resize(Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT)
        self.setup_ui()
    
    def setup_ui(self):
        # Main widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Log history text editor (read-only)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(80)
        self.log_text.setMaximumHeight(120)
        self.log_text.append("Select an output folder and enter a project name to begin")
        main_layout.addWidget(self.log_text)
        
        # Create a horizontal layout for the main viewing area
        view_area = QHBoxLayout()
        main_layout.addLayout(view_area)
        
        # Graphics view for displaying the video frame and mask (LEFT SIDE)
        graphics_section = QVBoxLayout()
        graphics_title = QLabel("Image Labeling")
        graphics_title.setAlignment(Qt.AlignCenter)
        
        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setMinimumSize(Config.FRAME_WIDTH, Config.FRAME_HEIGHT)
        self.graphics_view.setMouseTracking(True)  # Enable mouse tracking for line preview
        
        # Add pixmap items to scene for frame and mask
        self.frame_pixmap_item = QGraphicsPixmapItem()
        self.mask_pixmap_item = QGraphicsPixmapItem()
        self.graphics_scene.addItem(self.frame_pixmap_item)
        self.graphics_scene.addItem(self.mask_pixmap_item)
        
        # Setup mouse events for the view
        self.graphics_view.mousePressEvent = self.on_view_mouse_press
        self.graphics_view.mouseMoveEvent = self.on_view_mouse_move
        self.graphics_view.setMouseTracking(True)
        
        # Preview elements for line drawing
        self.last_point = None
        self.preview_line = None
        
        graphics_section.addWidget(graphics_title)
        graphics_section.addWidget(self.graphics_view)
        
        # First add the graphics view to the main view area (LEFT)
        view_area.addLayout(graphics_section)
        
        # Frame list with title (RIGHT SIDE)
        frame_list_section = QVBoxLayout()
        frame_list_title = QLabel("Frame List")
        frame_list_title.setAlignment(Qt.AlignCenter)
        self.frame_list = QListWidget()
        self.frame_list.setMinimumWidth(150)
        self.frame_list.itemClicked.connect(self.on_frame_list_item_clicked)
        
        frame_list_section.addWidget(frame_list_title)
        frame_list_section.addWidget(self.frame_list)
        
        # Then add the frame list to the main view area (RIGHT)
        view_area.addLayout(frame_list_section)
        
        # Controls and settings
        controls_layout = QHBoxLayout()
        
        # Line thickness control
        thickness_group = QGroupBox("Line Thickness")
        thickness_layout = QHBoxLayout()
        
        self.thickness_spinbox = QSpinBox()
        self.thickness_spinbox.setMinimum(Config.MIN_LINE_THICKNESS)
        self.thickness_spinbox.setMaximum(Config.MAX_LINE_THICKNESS)
        self.thickness_spinbox.setValue(Config.DEFAULT_LINE_THICKNESS)
        self.thickness_spinbox.valueChanged.connect(self.on_thickness_changed)
        thickness_layout.addWidget(self.thickness_spinbox)
        thickness_group.setLayout(thickness_layout)
        controls_layout.addWidget(thickness_group)
        
        # Additional controls can be added here if needed
        # Status label removed - using log_text instead
        
        main_layout.addLayout(controls_layout)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        # Previous button
        self.prev_button = QPushButton("Previous")
        self.prev_button.setFixedSize(Config.BUTTON_WIDTH, Config.BUTTON_HEIGHT)
        self.prev_button.clicked.connect(self.previous_frame_requested.emit)
        self.prev_button.setEnabled(False)
        button_layout.addWidget(self.prev_button)
        
        # Next button
        self.next_button = QPushButton("Next")
        self.next_button.setFixedSize(Config.BUTTON_WIDTH, Config.BUTTON_HEIGHT)
        self.next_button.clicked.connect(self.next_frame_requested.emit)
        self.next_button.setEnabled(False)
        button_layout.addWidget(self.next_button)
        
        # Undo button
        self.undo_button = QPushButton("Undo")
        self.undo_button.setFixedSize(Config.BUTTON_WIDTH, Config.BUTTON_HEIGHT)
        self.undo_button.clicked.connect(self.undo_requested.emit)
        self.undo_button.setEnabled(False)
        button_layout.addWidget(self.undo_button)
        
        # Save button
        self.save_button = QPushButton("Save")
        self.save_button.setFixedSize(Config.BUTTON_WIDTH, Config.BUTTON_HEIGHT)
        self.save_button.clicked.connect(self.save_requested.emit)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)
        
        # Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.setFixedSize(Config.BUTTON_WIDTH, Config.BUTTON_HEIGHT)
        self.reset_button.clicked.connect(self.reset_requested.emit)
        self.reset_button.setEnabled(False)
        button_layout.addWidget(self.reset_button)
        
        # Visualize button
        self.visualize_button = QPushButton("Visualize")
        self.visualize_button.setFixedSize(Config.BUTTON_WIDTH, Config.BUTTON_HEIGHT)
        self.visualize_button.clicked.connect(self.visualize_requested.emit)
        self.visualize_button.setEnabled(False)
        button_layout.addWidget(self.visualize_button)

        # Create Dataset button
        self.dataset_button = QPushButton("Create Dataset")
        self.dataset_button.setFixedSize(Config.BUTTON_WIDTH, Config.BUTTON_HEIGHT)
        self.dataset_button.clicked.connect(self.create_dataset_requested.emit)
        self.dataset_button.setEnabled(False)
        button_layout.addWidget(self.dataset_button)
        
        # Load video button
        self.load_video_button = QPushButton("Load Video")
        self.load_video_button.setFixedSize(Config.BUTTON_WIDTH, Config.BUTTON_HEIGHT)
        self.load_video_button.clicked.connect(self.on_load_video_clicked)
        self.load_video_button.setEnabled(False)
        button_layout.addWidget(self.load_video_button)
        
        main_layout.addLayout(button_layout)
        self.setCentralWidget(central_widget)
    
    def on_load_video_clicked(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        
        if video_path:
            self.video_file_selected.emit(video_path)
    
    def prompt_for_output_folder(self):
        output_folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if output_folder:
            self.output_folder_selected.emit(output_folder)
            return True
        return False
    
    def prompt_for_project_name(self):
        project_name, ok = QInputDialog.getText(self, "Project Name", "Enter project name:")
        if ok and project_name:
            self.project_name_entered.emit(project_name)
            return True
        return False
    
    def enable_video_loading(self):
        self.load_video_button.setEnabled(True)
    
    def update_frame_display(self, pixmap):
        self.frame_pixmap_item.setPixmap(pixmap)
        self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.KeepAspectRatio)
        
    def update_mask_display(self, pixmap):
        self.mask_pixmap_item.setPixmap(pixmap)
        # No need to call fitInView again as it's the same view
        

    def on_view_mouse_press(self, event):
        scene_pos = self.graphics_view.mapToScene(event.pos())
        current_point = QPoint(int(scene_pos.x()), int(scene_pos.y()))
        
        if self.last_point is None:
            # First point of line
            self.last_point = current_point
            self.update_status("First point selected. Click again to draw line.")
        else:
            # Second point of line, emit signal to draw the line
            self.points_selected.emit(self.last_point, current_point)
            
            # Remove preview line if exists
            if self.preview_line and self.preview_line in self.graphics_scene.items():
                self.graphics_scene.removeItem(self.preview_line)
                self.preview_line = None
            
            self.last_point = None
            self.update_status("Line drawn. Click to draw another line.")
            
            # Return focus to main window after drawing line
            self.setFocus()
    
    def on_view_mouse_move(self, event):
        if self.last_point is not None:
            # If we have a first point, show preview line to current mouse position
            scene_pos = self.graphics_view.mapToScene(event.pos())
            current_point = QPoint(int(scene_pos.x()), int(scene_pos.y()))
            
            # Remove previous preview line if exists
            if self.preview_line and self.preview_line in self.graphics_scene.items():
                self.graphics_scene.removeItem(self.preview_line)
            
            # Create new preview line with transparency
            pen = QPen(QColor(0, 255, 0, 128))  # Green color with 50% transparency
            pen.setWidth(self.thickness_spinbox.value())
            
            self.preview_line = QGraphicsLineItem(self.last_point.x(), self.last_point.y(), 
                                                current_point.x(), current_point.y())
            self.preview_line.setPen(pen)
            self.graphics_scene.addItem(self.preview_line)
    
    def on_thickness_changed(self, value):
        self.line_thickness_changed.emit(value)
    def clear_frame_list(self):
        # Clear the frame list when loading a new video
        self.frame_list.clear()
    
    def update_status(self, text):
        # Append message to log text editor instead of setting label text
        self.log_text.append(text)
    
    def enable_navigation_buttons(self, prev_enabled, next_enabled, save_enabled=False):
        self.prev_button.setEnabled(prev_enabled)
        self.next_button.setEnabled(next_enabled)
        self.save_button.setEnabled(save_enabled)
        self.reset_button.setEnabled(save_enabled)  # Reset is enabled when frame is loaded (same condition as save)
        self.visualize_button.setEnabled(save_enabled)
        self.dataset_button.setEnabled(save_enabled)
    
    def show_error(self, title, message):
        QMessageBox.critical(self, title, message)
    
    def show_info(self, title, message):
        QMessageBox.information(self, title, message)
    
    def on_frame_list_item_clicked(self, item):
        # Extract frame index from the item text and emit the signal
        try:
            # Item text format: "Frame X" where X is now already 0-based
            frame_text = item.text()
            # Extract the number part directly (no need to subtract 1 anymore)
            frame_index = int(frame_text.split(' ')[1])
            self.frame_selected.emit(frame_index)
            
            # Return focus to the main window after selection to ensure keyboard shortcuts work
            self.setFocus()
        except (ValueError, IndexError):
            # Handle any parsing errors
            self.show_error("Error", "Invalid frame selection.")
    
    def add_frame_to_list(self, frame_index):
        # Add a new frame to the list widget
        frame_text = f"Frame {frame_index}"
        self.frame_list.addItem(frame_text)
    
    def keyPressEvent(self, event):
        # Handle keyboard shortcuts
        key = event.key()
        
        if key == Qt.Key_S:  # S for save
            if self.save_button.isEnabled():
                self.save_requested.emit()
        elif key == Qt.Key_N:  # N for next
            if self.next_button.isEnabled():
                self.next_frame_requested.emit()
        elif key == Qt.Key_B:  # B for previous
            if self.prev_button.isEnabled():
                self.previous_frame_requested.emit()
        elif key == Qt.Key_R:  # R for reset
            if self.reset_button.isEnabled():
                self.reset_requested.emit()
        elif key == Qt.Key_Z:  # Z for undo
            if self.undo_button.isEnabled():
                self.undo_requested.emit()
        else:
            # Pass the event to the parent class for default handling
            super().keyPressEvent(event)
