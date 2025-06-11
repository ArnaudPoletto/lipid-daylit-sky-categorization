#!/usr/bin/env python3
"""
Sky Finder Active Learning Tool

This script helps with active learning by displaying images sorted by model confidence
(based on entropy values), showing both continuous and binary predictions overlaid on the image.

Usage:
    python sky_finder_active_learning.py [--width WIDTH] [--height HEIGHT]

Options:
    --width WIDTH    Window width (default: 1200)
    --height HEIGHT  Window height (default: 800)

Key controls:
    y: Mark current image as good for training (Yes)
    n: Mark current image as not good for training (No)
    s: Skip current image (will be shown again later)
    Delete: Undo the last classification
    q: Quit the application
"""

import os
import sys
import shutil
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import glob
from collections import deque
import json
import argparse

# Constants
CACHE_SIZE = 100
CATEGORIES = ['clear', 'partial', 'overcast']
CACHE_FILE = 'active_learning_cache.json'
CONFIG_FILE = 'active_learning_config.json'

# Check if the script is in the same directory as sky_finder_active_learning
# or if sky_finder_active_learning is a relative path
if os.path.exists('sky_finder_active_learning'):
    BASE_PATH = 'sky_finder_active_learning'
else:
    # Try current directory (in case script is placed inside the directory)
    BASE_PATH = '.'

print(f"Using base path: {BASE_PATH}")

SPLIT_PATH = os.path.join(BASE_PATH, 'train')
IMAGES_PATH = os.path.join(SPLIT_PATH, 'images')
ENTROPY_PATH = os.path.join(SPLIT_PATH, 'entropies')
BINARY_PRED_PATH = os.path.join(SPLIT_PATH, 'binary_predictions')
CONTINUOUS_PRED_PATH = os.path.join(SPLIT_PATH, 'continuous_predictions')

# Output paths for classified images
KEEP_PATH = os.path.join(BASE_PATH, 'keep')
DISCARD_PATH = os.path.join(BASE_PATH, 'discard')

# Configurable parameters
WINDOW_WIDTH = 2800
WINDOW_HEIGHT = 800

class SkyFinderActiveLearningTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Sky Finder Active Learning Tool")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        
        # Load configuration or set defaults
        self.load_config()
        
        # Initialize cache
        self.operation_cache = deque(maxlen=CACHE_SIZE)
        self.load_cache()
        
        # Image data structures
        self.image_data = []  # List of dicts with image info and entropy score
        self.current_index = 0
        
        # Setup UI
        self.setup_ui()
        
        # Store already classified images
        self.classified_images = set()
        self.load_classified_images()
        
        # Load images data
        self.load_image_data()

    def load_config(self):
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    self.current_index = config.get('current_index', 0)
            else:
                self.current_index = 0
        except Exception as e:
            print(f"Error loading config: {e}")
            self.current_index = 0

    def save_config(self):
        try:
            config = {
                'current_index': self.current_index
            }
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f)
        except Exception as e:
            print(f"Error saving config: {e}")

    def load_cache(self):
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, 'r') as f:
                    cache_data = json.load(f)
                    for operation in cache_data:
                        self.operation_cache.append(operation)
        except Exception as e:
            print(f"Error loading cache: {e}")

    def save_cache(self):
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump(list(self.operation_cache), f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def load_classified_images(self):
        """Load all images that have already been classified (kept or discarded)"""
        self.classified_images = set()
        
        # Check both KEEP and DISCARD directories
        for base_dir in [KEEP_PATH, DISCARD_PATH]:
            if not os.path.exists(base_dir):
                continue
            
            # Walk through all categories and camera IDs
            for category in CATEGORIES:
                category_path = os.path.join(base_dir, category)
                if not os.path.exists(category_path):
                    continue
                
                # For each camera ID
                for camera_id in os.listdir(category_path):
                    camera_path = os.path.join(category_path, camera_id)
                    if not os.path.isdir(camera_path):
                        continue
                    
                    # Get all image files
                    for file in os.listdir(camera_path):
                        # Only track the original images (not binary/continuous/entropy versions)
                        if not file.startswith(('binary_', 'continuous_', 'entropy_')) and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            # Create a unique identifier for the image: category/camera_id/filename
                            image_key = f"{category}/{camera_id}/{file}"
                            self.classified_images.add(image_key)
        
        print(f"Found {len(self.classified_images)} already classified images")

    def setup_ui(self):
        # Main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Info frame at top
        info_frame = tk.Frame(self.main_frame)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Image filename and path
        self.filename_label = tk.Label(info_frame, text="File: N/A", font=("Arial", 12, "bold"))
        self.filename_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Progress frame under filename
        progress_frame = tk.Frame(info_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Info labels in a grid layout 
        self.progress_label = tk.Label(progress_frame, text="0/0 images classified", font=("Arial", 12))
        self.progress_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        
        self.confidence_label = tk.Label(progress_frame, text="Confidence: N/A", font=("Arial", 12))
        self.confidence_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        self.category_label = tk.Label(progress_frame, text="Prediction: N/A", font=("Arial", 12))
        self.category_label.grid(row=0, column=2, sticky=tk.W, padx=(0, 20))
        
        # Camera ID info
        self.camera_label = tk.Label(progress_frame, text="Camera ID: N/A", font=("Arial", 12))
        self.camera_label.grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        
        # Add a label for displaying the count of already processed images
        self.processed_label = tk.Label(progress_frame, text="Already processed: 0", font=("Arial", 12))
        self.processed_label.grid(row=0, column=4, sticky=tk.W, padx=(0, 20))
        
        # Images container frame - horizontal layout for main image and continuous prediction
        images_container = tk.Frame(self.main_frame)
        images_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        # Configure grid weights for resizing (equal space for all three images)
        images_container.columnconfigure(0, weight=1)  # Original image
        images_container.columnconfigure(1, weight=1)  # Binary overlay
        images_container.columnconfigure(2, weight=1)  # Continuous prediction
        images_container.rowconfigure(0, weight=1)
        
        # Original image frame (left panel)
        self.original_frame = tk.Frame(images_container, bd=2, relief=tk.GROOVE)
        self.original_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
        
        # Title for original image
        tk.Label(self.original_frame, text="Original Image", font=("Arial", 12, "bold")).pack(pady=(5, 5))
        
        # Image label for original image
        self.original_label = tk.Label(self.original_frame)
        self.original_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Binary overlay image frame (middle panel)
        self.binary_frame = tk.Frame(images_container, bd=2, relief=tk.GROOVE)
        self.binary_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Title for binary overlay
        tk.Label(self.binary_frame, text="Binary Prediction Overlay", font=("Arial", 12, "bold")).pack(pady=(5, 5))
        
        # Image label for binary overlay
        self.binary_label = tk.Label(self.binary_frame)
        self.binary_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Continuous prediction frame (right panel)
        self.continuous_frame = tk.Frame(images_container, bd=2, relief=tk.GROOVE)
        self.continuous_frame.grid(row=0, column=2, sticky="nsew", padx=(5, 0), pady=5)
        
        # Title for continuous prediction
        tk.Label(self.continuous_frame, text="Continuous Prediction", font=("Arial", 12, "bold")).pack(pady=(5, 5))
        
        # Label for continuous prediction image
        self.continuous_label = tk.Label(self.continuous_frame)
        self.continuous_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Buttons frame
        button_frame = tk.Frame(self.main_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Buttons
        self.keep_button = tk.Button(
            button_frame, 
            text="Keep (y)", 
            command=lambda: self.classify_image("keep"),
            font=("Arial", 11),
            width=15,
            height=2,
            bg="#e6ffe6"  # Light green
        )
        self.keep_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.discard_button = tk.Button(
            button_frame, 
            text="Discard (n)", 
            command=lambda: self.classify_image("discard"),
            font=("Arial", 11),
            width=15,
            height=2,
            bg="#ffe6e6"  # Light red
        )
        self.discard_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.skip_button = tk.Button(
            button_frame, 
            text="Skip (s)", 
            command=self.skip_image,
            font=("Arial", 11),
            width=15,
            height=2,
            bg="#e6e6ff"  # Light blue
        )
        self.skip_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.undo_button = tk.Button(
            button_frame, 
            text="Undo (Delete)", 
            command=self.undo_last_operation,
            font=("Arial", 11),
            width=15,
            height=2
        )
        self.undo_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.quit_button = tk.Button(
            button_frame, 
            text="Quit (q)", 
            command=self.quit_application,
            font=("Arial", 11),
            width=15,
            height=2
        )
        self.quit_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Status bar
        self.status_bar = tk.Label(self.main_frame, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
        
        # Bind keyboard shortcuts
        self.root.bind('<KeyPress-y>', lambda e: self.classify_image("keep"))
        self.root.bind('<KeyPress-n>', lambda e: self.classify_image("discard"))
        self.root.bind('<KeyPress-s>', lambda e: self.skip_image())
        self.root.bind('<Delete>', lambda e: self.undo_last_operation())
        self.root.bind('<KeyPress-q>', lambda e: self.quit_application())

    def load_image_data(self):
        """Load all images and their entropy values"""
        self.status_bar.config(text="Loading image data...")
        self.root.update()
        
        self.image_data = []
        
        # Create output directories if they don't exist
        os.makedirs(KEEP_PATH, exist_ok=True)
        os.makedirs(DISCARD_PATH, exist_ok=True)
        
        # Debug output to help diagnose issues
        print(f"Looking for images in: {IMAGES_PATH}")
        if not os.path.exists(IMAGES_PATH):
            print(f"WARNING: {IMAGES_PATH} does not exist!")
            self.status_bar.config(text=f"Error: {IMAGES_PATH} not found!")
            return
            
        # Use os.walk to find all image files in the images directory
        processed_count = 0
        skipped_count = 0
        
        for category in CATEGORIES:
            category_images_path = os.path.join(IMAGES_PATH, category)
            if not os.path.exists(category_images_path):
                print(f"Category path does not exist: {category_images_path}")
                continue
                
            print(f"Processing category: {category}")
            
            # Walk through all subdirectories in this category
            for root, dirs, files in os.walk(category_images_path):
                for filename in files:
                    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                        
                    # Extract camera_id from the path
                    # The path structure is .../images/category/camera_id/filename
                    path_parts = root.split(os.sep)
                    if len(path_parts) < 2:
                        continue
                        
                    camera_id = path_parts[-1]  # Last part of the path is camera_id
                    
                    # Check if this image has already been classified
                    image_key = f"{category}/{camera_id}/{filename}"
                    if image_key in self.classified_images:
                        skipped_count += 1
                        continue
                    
                    # Construct paths for all required files
                    img_path = os.path.join(root, filename)
                    binary_pred_path = os.path.join(BINARY_PRED_PATH, category, camera_id, filename)
                    continuous_pred_path = os.path.join(CONTINUOUS_PRED_PATH, category, camera_id, filename)
                    entropy_path = os.path.join(ENTROPY_PATH, category, camera_id, filename)
                    
                    # Debug output for the first file found
                    if len(self.image_data) == 0:
                        print(f"Sample file paths:")
                        print(f"Image: {img_path} - Exists: {os.path.exists(img_path)}")
                        print(f"Binary: {binary_pred_path} - Exists: {os.path.exists(binary_pred_path)}")
                        print(f"Continuous: {continuous_pred_path} - Exists: {os.path.exists(continuous_pred_path)}")
                        print(f"Entropy: {entropy_path} - Exists: {os.path.exists(entropy_path)}")
                    
                    # Skip if any of the required files don't exist
                    if not all(os.path.exists(p) for p in [img_path, binary_pred_path, continuous_pred_path, entropy_path]):
                        continue
                    
                    try:
                        # Calculate average entropy (lower means more confident)
                        entropy_img = Image.open(entropy_path)
                        entropy_array = np.array(entropy_img)
                        avg_entropy = np.mean(entropy_array)
                        
                        # Add to our dataset
                        self.image_data.append({
                            'img_path': img_path,
                            'binary_pred_path': binary_pred_path,
                            'continuous_pred_path': continuous_pred_path,
                            'entropy_path': entropy_path,
                            'category': category,
                            'camera_id': camera_id,
                            'filename': filename,
                            'entropy_score': avg_entropy,
                            'image_key': image_key
                        })
                        processed_count += 1
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
        
        print(f"Total images to process: {len(self.image_data)}")
        print(f"Already processed images: {skipped_count}")
        
        # Sort by entropy score (lowest/most confident first)
        self.image_data.sort(key=lambda x: x['entropy_score'])
        
        # Update processed label
        self.processed_label.config(text=f"Already processed: {skipped_count}")
        
        # Update progress label
        self.progress_label.config(text=f"0/{len(self.image_data)} images classified")
        
        if not self.image_data:
            self.status_bar.config(text="No new images found to classify!")
            return
            
        # Load the first image
        self.load_current_image()

    def load_current_image(self):
        """Load and display the three images: original, binary overlay, and continuous prediction"""
        if not self.image_data or self.current_index >= len(self.image_data):
            self.status_bar.config(text="No more images to classify")
            self.original_label.config(image=None)
            self.binary_label.config(image=None)
            self.continuous_label.config(image=None)
            self.filename_label.config(text="File: No more images")
            return
        
        current_data = self.image_data[self.current_index]
        
        try:
            # Load the original image
            img = Image.open(current_data['img_path'])
            
            # Load the binary prediction
            binary_pred = Image.open(current_data['binary_pred_path'])
            
            # Load the continuous prediction
            continuous_pred = Image.open(current_data['continuous_pred_path'])
            
            # Make sure all images have the same base size
            base_size = img.size
            
            # Resize predictions to match base image if needed
            if binary_pred.size != base_size:
                binary_pred = binary_pred.resize(base_size, Image.LANCZOS)
            if continuous_pred.size != base_size:
                continuous_pred = continuous_pred.resize(base_size, Image.LANCZOS)
            
            # 1. Prepare the original image
            original_img = img.copy()
            
            # 2. Create composite image with binary prediction overlay
            # Create an overlay with binary prediction (semi-transparent)
            binary_overlay = Image.new('RGBA', base_size, (0, 0, 0, 0))
            binary_overlay.paste(binary_pred.convert('RGBA'), (0, 0))
            
            # Make binary overlay semi-transparent
            binary_pixels = binary_overlay.load()
            for i in range(binary_overlay.width):
                for j in range(binary_overlay.height):
                    r, g, b, a = binary_pixels[i, j]
                    binary_pixels[i, j] = (r, g, b, 128)  # 50% transparency
            
            # Convert original to RGBA to support transparency
            composite = img.copy()
            if composite.mode != 'RGBA':
                composite = composite.convert('RGBA')
            
            # Apply binary overlay
            composite = Image.alpha_composite(composite, binary_overlay)
            
            # Calculate available space for each image (1/3 of window width)
            panel_width = (WINDOW_WIDTH - 60) // 3  # Account for padding and borders
            panel_height = WINDOW_HEIGHT - 200
            
            # Define a function to resize images proportionally
            def resize_for_display(image, width, height):
                img_ratio = image.width / image.height
                display_ratio = width / height
                
                if img_ratio > display_ratio:
                    # Image is wider than display area
                    new_width = width
                    new_height = int(width / img_ratio)
                else:
                    # Image is taller than display area
                    new_height = height
                    new_width = int(height * img_ratio)
                
                return image.resize((new_width, new_height), Image.LANCZOS)
            
            # Resize all images for display
            display_original = resize_for_display(original_img, panel_width, panel_height)
            display_composite = resize_for_display(composite, panel_width, panel_height)
            display_continuous = resize_for_display(continuous_pred, panel_width, panel_height)
            
            # Convert all to PhotoImage for Tkinter
            original_tk = ImageTk.PhotoImage(display_original)
            composite_tk = ImageTk.PhotoImage(display_composite)
            continuous_tk = ImageTk.PhotoImage(display_continuous)
            
            # Update the images
            self.original_label.config(image=original_tk)
            self.original_label.image = original_tk  # Keep reference
            
            self.binary_label.config(image=composite_tk)
            self.binary_label.image = composite_tk  # Keep reference
            
            self.continuous_label.config(image=continuous_tk)
            self.continuous_label.image = continuous_tk  # Keep reference
            
            # Update information labels at the top
            self.filename_label.config(text=f"File: {current_data['filename']}")
            self.progress_label.config(text=f"{self.current_index + 1}/{len(self.image_data)} images")
            self.confidence_label.config(text=f"Confidence: {1.0 - current_data['entropy_score']:.4f}")
            self.category_label.config(text=f"Prediction: {current_data['category']}")
            self.camera_label.config(text=f"Camera ID: {current_data['camera_id']}")
            
            # Update status
            self.status_bar.config(text=f"Loaded image: {current_data['filename']}")
                
        except Exception as e:
            self.status_bar.config(text=f"Error loading image: {e}")
            print(f"Error loading image: {e}")
            self.original_label.config(image=None)
            self.binary_label.config(image=None)
            self.continuous_label.config(image=None)

    def classify_image(self, decision):
        """Classify current image as keep or discard"""
        if not self.image_data or self.current_index >= len(self.image_data):
            self.status_bar.config(text="No image to classify")
            return
        
        current_data = self.image_data[self.current_index]
        
        # Create appropriate directory structure in output folder
        if decision == "keep":
            dest_dir = os.path.join(KEEP_PATH, current_data['category'], current_data['camera_id'])
            self.status_bar.config(text=f"Keeping image: {current_data['filename']}")
        else:  # discard
            dest_dir = os.path.join(DISCARD_PATH, current_data['category'], current_data['camera_id'])
            self.status_bar.config(text=f"Discarding image: {current_data['filename']}")
        
        os.makedirs(dest_dir, exist_ok=True)
        
        # Destination paths
        img_dest = os.path.join(dest_dir, current_data['filename'])
        binary_dest = os.path.join(dest_dir, f"binary_{current_data['filename']}")
        continuous_dest = os.path.join(dest_dir, f"continuous_{current_data['filename']}")
        entropy_dest = os.path.join(dest_dir, f"entropy_{current_data['filename']}")
        
        # Store information for undo operation
        operation = {
            'decision': decision,
            'data': current_data,
            'destinations': {
                'img': img_dest,
                'binary': binary_dest,
                'continuous': continuous_dest,
                'entropy': entropy_dest
            }
        }
        
        # Copy files to destination (using copy to preserve originals)
        try:
            shutil.copy2(current_data['img_path'], img_dest)
            shutil.copy2(current_data['binary_pred_path'], binary_dest)
            shutil.copy2(current_data['continuous_pred_path'], continuous_dest)
            shutil.copy2(current_data['entropy_path'], entropy_dest)
            
            # Add to operation cache
            self.operation_cache.append(operation)
            self.save_cache()
            
            # Add to classified images set
            self.classified_images.add(current_data['image_key'])
            
            # Move to next image
            self.current_index += 1
            self.save_config()
            
            # Load next image
            self.load_current_image()
            
        except Exception as e:
            self.status_bar.config(text=f"Error during classification: {e}")

    def skip_image(self):
        """Skip the current image and move to the next one"""
        if not self.image_data or self.current_index >= len(self.image_data):
            self.status_bar.config(text="No image to skip")
            return
        
        current_data = self.image_data[self.current_index]
        self.status_bar.config(text=f"Skipped image: {current_data['filename']}")
        
        # Move current image to the end of the list
        skipped_image = self.image_data.pop(self.current_index)
        self.image_data.append(skipped_image)
        
        # No need to update current_index as we removed the current item
        
        # Load the next image
        self.load_current_image()

    def undo_last_operation(self):
        """Undo the last classification operation"""
        if not self.operation_cache:
            self.status_bar.config(text="Nothing to undo")
            return
        
        # Get the last operation
        last_operation = self.operation_cache.pop()
        self.save_cache()
        
        try:
            # Remove the copied files
            for dest_type, dest_path in last_operation['destinations'].items():
                if os.path.exists(dest_path):
                    os.remove(dest_path)
            
            # Remove from classified images set
            image_key = last_operation['data'].get('image_key')
            if image_key and image_key in self.classified_images:
                self.classified_images.remove(image_key)
            
            # If the undone operation was for the image right before the current one,
            # we need to decrement the current_index
            if self.current_index > 0:
                self.current_index -= 1
                self.save_config()
            
            # Reload the current image
            self.load_current_image()
            
            self.status_bar.config(text="Last operation undone")
            
        except Exception as e:
            self.status_bar.config(text=f"Error during undo: {e}")

    def quit_application(self):
        """Quit the application"""
        self.save_config()
        self.save_cache()
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Sky Finder Active Learning Tool')
    parser.add_argument('--width', type=int, default=WINDOW_WIDTH, help='Window width')
    parser.add_argument('--height', type=int, default=WINDOW_HEIGHT, help='Window height')
    parser.add_argument('--split', type=str, choices=['train', 'val'], default='train', help='Split to use (train or val)')
    
    args = parser.parse_args()
    
    # Update globals from command line
    WINDOW_WIDTH = args.width
    WINDOW_HEIGHT = args.height
    SPLIT_PATH = os.path.join(BASE_PATH, args.split)
    IMAGES_PATH = os.path.join(SPLIT_PATH, 'images')
    ENTROPY_PATH = os.path.join(SPLIT_PATH, 'entropies')
    BINARY_PRED_PATH = os.path.join(SPLIT_PATH, 'binary_predictions')
    CONTINUOUS_PRED_PATH = os.path.join(SPLIT_PATH, 'continuous_predictions')
    KEEP_PATH = os.path.join(BASE_PATH, f'{args.split}_keep')
    DISCARD_PATH = os.path.join(BASE_PATH, f'{args.split}_discard')
    
    root = tk.Tk()
    app = SkyFinderActiveLearningTool(root)
    root.mainloop()