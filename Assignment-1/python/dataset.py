import os
import cv2  # Allowed ONLY for image reading
import deep_framework as df # Imports your C++ backend
import random
import time

class ImageFolder:
    def __init__(self, root_dir, resize_shape=(32, 32)):
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        self.samples = [] 
        self.class_to_idx = {}
        self.load_time = 0.0
        
        # Measure loading time as required by assignment
        t0 = time.time()
        
        # 1. Infer labels from folder names
        if not os.path.exists(root_dir):
            raise ValueError(f"Directory not found: {root_dir}")
            
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        print(f"Found classes: {classes}")
        
        for idx, cls_name in enumerate(classes):
            self.class_to_idx[cls_name] = idx
            cls_folder = os.path.join(root_dir, cls_name)
            
            # Gather all valid images
            for img_name in os.listdir(cls_folder):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_folder, img_name), idx))
        
        self.load_time = time.time() - t0
        print(f"Dataset loaded: {len(self.samples)} images in {self.load_time:.4f}s")

    def shuffle(self):
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def get_batch(self, start_idx, batch_size):
        end_idx = min(start_idx + batch_size, len(self.samples))
        batch_samples = self.samples[start_idx:end_idx]
        actual_bs = len(batch_samples)

        # We use flat lists because we cannot use NumPy
        flat_pixels = []
        flat_labels = []

        h, w = self.resize_shape
        
        for path, label in batch_samples:
            # Load and Resize
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: 
                # Handle broken images by padding with zeros (simplest safe fallback)
                flat_pixels.extend([0.0] * (h*w))
                flat_labels.append(float(label))
                continue

            img = cv2.resize(img, (w, h))
            
            # Normalize 0-255 -> 0.0-1.0
            # Convert directly to list of floats
            pixels = [p / 255.0 for p in img.flatten()]
            flat_pixels.extend(pixels)
            flat_labels.append(float(label))

        # Create C++ Tensors with correct shape (B, C, H, W)
        # Note: 1 Channel (Grayscale)
        x = df.create_tensor([actual_bs, 1, h, w], False)
        y = df.create_tensor([actual_bs], False)
        
        # Fill data into C++ Tensor
        x.data = flat_pixels
        y.data = flat_labels
        
        return x, y