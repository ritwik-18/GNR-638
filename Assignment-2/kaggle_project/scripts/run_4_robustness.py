import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.models import get_model, setup_finetuning_strategy
from src.utils import evaluate
from src.corruptions import get_corrupted_val_transform

class TransformSubset(Subset):
    """Wrapper to apply specific transforms to a subset."""
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform
        
    def __getitem__(self, idx):
        img, label = self.dataset.samples[self.indices[idx]]
        from PIL import Image
        img = Image.open(img).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def get_eval_loader(data_dir, corruption_type, severity=None, batch_size=32, seed=37):
    """
    Recreates the exact 20% validation split used in training, 
    but applies the specified corruption transform pipeline.
    """
    full_dataset = datasets.ImageFolder(root=data_dir)
    targets = full_dataset.targets
    indices = np.arange(len(targets))
    
    # Recreate the exact same 80/20 split using the global seed
    _, val_idx = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=targets
    )
    
    # Get the corruption pipeline
    transform = get_corrupted_val_transform(corruption_type, severity)
    
    # Wrap in our subset class
    val_dataset = TransformSubset(full_dataset, val_idx, transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return val_loader, len(full_dataset.classes)

def main():
    print("="*60)
    print("Starting Scenario 4.4: Corruption Robustness Evaluation")
    print("="*60)
    
    # Define the corruption scenarios to test 
    corruptions = [
        {"name": "Clean (Baseline)", "type": "clean", "severity": None},
        {"name": "Gaussian Noise (s=0.05)", "type": "gaussian_noise", "severity": 0.05},
        {"name": "Gaussian Noise (s=0.1)", "type": "gaussian_noise", "severity": 0.1},
        {"name": "Gaussian Noise (s=0.2)", "type": "gaussian_noise", "severity": 0.2},
        {"name": "Motion Blur", "type": "motion_blur", "severity": 15}, # Kernel size 15
        {"name": "Brightness Shift", "type": "brightness_shift", "severity": 1.5} # 1.5x brighter
    ]
    
    results = {}
    criterion = nn.CrossEntropyLoss()
    
    for model_name in config.SELECTED_MODELS:
        print(f"\n>>> Evaluating Model: {model_name.upper()} <<<")
        results[model_name] = {}
        
        # We will load the model weights from the Linear Probe scenario as our baseline.
        # Make sure you have run run_01_linear_probe.py first!
        model_weights_path = os.path.join(config.MODELS_DIR, f"{model_name}_linear_probe.pth")
        
        if not os.path.exists(model_weights_path):
            print(f"WARNING: Weights not found for {model_name} at {model_weights_path}.")
            print("Skipping... Please run Scenario 4.1 first.")
            continue
            
        # Initialize model and load trained weights
        # We need num_classes=30 to match the saved weights
        model = get_model(model_name, num_classes=config.NUM_CLASSES, pretrained=False) 
        model = setup_finetuning_strategy(model, model_name, strategy="linear_probe")
        model.load_state_dict(torch.load(model_weights_path, map_location=config.DEVICE))
        model = model.to(config.DEVICE)
        model.eval()
        
        clean_acc = None
        
        for corr in corruptions:
            print(f"Testing under: {corr['name']}...")
            
            # Get dataloader with specific corruption applied at evaluation time [cite: 87]
            val_loader, _ = get_eval_loader(
                data_dir=config.DATA_DIR, 
                corruption_type=corr['type'], 
                severity=corr['severity'],
                batch_size=config.BATCH_SIZE,
                seed=config.SEED
            )
            
            # Evaluate
            _, acc, _, _ = evaluate(model, val_loader, criterion, config.DEVICE)
            
            # Store clean accuracy to calculate Relative Robustness later
            if corr['type'] == 'clean':
                clean_acc = acc
                
            # Calculate metrics [cite: 89-94]
            corruption_error = 1.0 - acc
            relative_robustness = acc / clean_acc if clean_acc else 1.0
            
            results[model_name][corr['name']] = {
                "acc": acc,
                "error": corruption_error,
                "rel_rob": relative_robustness
            }
            
    # Print Final Summary Table [cite: 20]
    print("\n" + "="*85)
    print("CORRUPTION ROBUSTNESS SUMMARY")
    print("="*85)
    print(f"{'Model':<15} | {'Corruption':<25} | {'Val Acc':<8} | {'Corr Error':<10} | {'Rel Robustness'}")
    print("-" * 85)
    
    for model_name in config.SELECTED_MODELS:
        if model_name not in results:
            continue
            
        for corr in corruptions:
            corr_name = corr['name']
            data = results[model_name][corr_name]
            print(f"{model_name:<15} | {corr_name:<25} | {data['acc']:<8.4f} | {data['error']:<10.4f} | {data['rel_rob']:.4f}")
        print("-" * 85)

if __name__ == "__main__":
    main()