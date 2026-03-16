import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Data Transforms
# ==========================================

def get_transforms(is_train=True):
    """
    Returns standard ImageNet transforms for pre-trained CNNs.
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# ==========================================
# 2. Main DataLoader & Few-Shot Logic
# ==========================================

def get_dataloaders(data_dir, batch_size=32, seed=37, train_fraction=1.0, num_workers=4):
    """
    Loads the AID dataset, creates a Train/Val split, and applies few-shot subsetting.
    """
    # Load dataset twice to assign proper transforms natively
    train_full_dataset = datasets.ImageFolder(root=data_dir, transform=get_transforms(is_train=True))
    val_full_dataset = datasets.ImageFolder(root=data_dir, transform=get_transforms(is_train=False))
    
    # Extract targets for stratified splitting
    targets = train_full_dataset.targets
    indices = np.arange(len(targets))
    
    # 1. Create Base Train (80%) and Val (20%) Split
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=targets
    )
    
    # 2. Apply Few-Shot Subsetting (Scenario 4.3)
    if train_fraction < 1.0:
        # Extract the targets of the training subset to maintain class balance
        train_targets = [targets[i] for i in train_idx]
        
        # Sub-sample the training indices
        train_idx, _ = train_test_split(
            train_idx, 
            train_size=train_fraction, 
            random_state=seed, 
            stratify=train_targets
        )
    
    # 3. Create PyTorch Subsets using the correctly transformed base datasets
    train_dataset = Subset(train_full_dataset, train_idx)
    val_dataset = Subset(val_full_dataset, val_idx)
    
    # 4. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, len(train_full_dataset.classes)

# ==========================================
# 3. Layer Probing Subset (Scenario 4.5)
# ==========================================

def get_layer_probing_loader(data_dir, batch_size=32, seed=37, num_workers=4):
    """
    Creates a fixed subset of exactly 30 classes with 30 samples per class 
    for PCA/Layer-wise probing (Scenario 4.5).
    """
    full_dataset = datasets.ImageFolder(root=data_dir, transform=get_transforms(is_train=False))
    targets = np.array(full_dataset.targets)
    
    probing_indices = []
    
    # Set seed for reproducibility across all model runs
    rng = np.random.default_rng(seed)
    
    # Iterate through all 30 classes
    for class_idx in range(len(full_dataset.classes)):
        # Find all image indices belonging to this class
        class_specific_indices = np.where(targets == class_idx)[0]
        
        # Randomly select exactly 30 samples for this class
        selected_indices = rng.choice(class_specific_indices, size=30, replace=False)
        probing_indices.extend(selected_indices)
        
    probing_dataset = Subset(full_dataset, probing_indices)
    
    # Shuffle=False is crucial here so embeddings map perfectly to labels across different models
    probing_loader = DataLoader(probing_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return probing_loader