import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.dataset import get_dataloaders
from src.models import get_model, setup_finetuning_strategy
from src.utils import train_model

def calculate_relative_drop(acc_100, acc_5):
    """Calculates the relative performance drop as defined in the assignment."""
    # Formula: (Acc_100% - Acc_5%) / Acc_100% 
    if acc_100 == 0:
        return 0.0
    return (acc_100 - acc_5) / acc_100

def main():
    print("="*50)
    print("Starting Scenario 4.3: Few-Shot Learning Analysis ")
    print("="*50)
    
    # We will track results to print a clean summary table at the end
    results_summary = {}
    
    fractions = {
        "100%": 1.0,
        "20%": 0.20,
        "5%": 0.05
    }
    
    for model_name in config.SELECTED_MODELS:
        print(f"\n" + "="*40)
        print(f"Evaluating Model: {model_name.upper()}")
        print("="*40)
        
        results_summary[model_name] = {}
        
        for fraction_name, fraction_val in fractions.items():
            print(f"\n--- Data Regime: {fraction_name} ({fraction_val * 100}%) ---")
            
            # Load Dataset with specific fraction and group seed [cite: 73]
            train_loader, val_loader, num_classes = get_dataloaders(
                data_dir=config.DATA_DIR,
                batch_size=config.BATCH_SIZE,
                seed=config.SEED,
                train_fraction=fraction_val 
            )
            
            # Initialize fresh model
            model = get_model(model_name, num_classes=num_classes, pretrained=True)
            
            # For few-shot, we standardly use linear probing or full fine-tuning. 
            # We will use linear probing here to prevent rapid overfitting on tiny datasets, 
            # but you can change this to "full" if you prefer.
            model = setup_finetuning_strategy(model, model_name, strategy="linear_probe")
            model = model.to(config.DEVICE)
            
            # Setup Optimizer
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.Adam(trainable_params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
            criterion = nn.CrossEntropyLoss()
            
            # Determine max epochs based on data size 
            num_epochs = config.MAX_EPOCHS_FULL if fraction_val == 1.0 else config.MAX_EPOCHS_FEW_SHOT
            
            # Train the model
            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=config.DEVICE,
                num_epochs=num_epochs,
                track_grads=False
            )
            
            # Extract final metrics (using the best validation epoch)
            best_epoch_idx = history['val_acc'].index(max(history['val_acc']))
            best_val_acc = history['val_acc'][best_epoch_idx]
            corresponding_train_acc = history['train_acc'][best_epoch_idx]
            train_val_gap = corresponding_train_acc - best_val_acc # 
            
            results_summary[model_name][fraction_name] = {
                "val_acc": best_val_acc,
                "train_acc": corresponding_train_acc,
                "gap": train_val_gap
            }
            
            print(f"Results for {fraction_name}: Val Acc = {best_val_acc:.4f}, Train-Val Gap = {train_val_gap:.4f}")
            
    # Print Final Summary Table [cite: 20]
    print("\n" + "="*60)
    print("FEW-SHOT LEARNING SUMMARY ")
    print("="*60)
    print(f"{'Model':<15} | {'Regime':<6} | {'Val Acc':<8} | {'Train-Val Gap':<13} | {'Rel. Drop (100%->5%)'}")
    print("-" * 60)
    
    for model_name in config.SELECTED_MODELS:
        acc_100 = results_summary[model_name]["100%"]["val_acc"]
        acc_5 = results_summary[model_name]["5%"]["val_acc"]
        rel_drop = calculate_relative_drop(acc_100, acc_5) # 
        
        for regime in ["100%", "20%", "5%"]:
            data = results_summary[model_name][regime]
            drop_str = f"{rel_drop:.4f}" if regime == "100%" else "" # Only print drop on the first line
            print(f"{model_name:<15} | {regime:<6} | {data['val_acc']:<8.4f} | {data['gap']:<13.4f} | {drop_str}")
            
    print("="*60)

if __name__ == "__main__":
    main()