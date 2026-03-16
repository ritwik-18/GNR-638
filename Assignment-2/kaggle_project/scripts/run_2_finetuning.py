import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.dataset import get_dataloaders
from src.models import get_model, setup_finetuning_strategy
from src.utils import train_model, calculate_efficiency_metrics
from src.visualizations import plot_finetuning_summary

def get_unfrozen_percentage(model):
    """Calculates the exact percentage of unfrozen parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    unfrozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (unfrozen_params / total_params) * 100

def plot_convergence_comparison(histories, model_name, save_path):
    """Plots training loss vs epoch for all strategies on a single graph."""
    plt.figure(figsize=(10, 6))
    for strategy_name, history in histories.items():
        epochs = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs, history['train_loss'], label=strategy_name, marker='o', markersize=4)
        
    plt.title(f'Convergence Stability Comparison - {model_name.upper()}')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_gradient_norms(grad_norms_dict, model_name, strategy_name, save_path):
    """Plots the average gradient norm across different layers."""
    # Filter to evenly spaced layers if there are too many (for readable plotting)
    layers = list(grad_norms_dict.keys())
    if len(layers) > 20:
        step = len(layers) // 20
        layers = layers[::step]
        
    norms = [grad_norms_dict[layer] for layer in layers]
    
    # Clean up layer names for the x-axis
    clean_labels = [l.replace('.weight', '').replace('.bias', '')[-15:] for l in layers]
    
    plt.figure(figsize=(12, 6))
    plt.bar(clean_labels, norms, color='coral')
    plt.title(f'Average Gradient Norms Across Layers - {model_name.upper()} ({strategy_name})')
    plt.xlabel('Layer (Truncated Names)')
    plt.ylabel('L2 Gradient Norm')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    print("="*50)
    print("Starting Scenario 4.2: Fine-Tuning Strategies")
    print("="*50)
    
    # 1. Load Dataset (100% of training data)
    print("Loading Aerial Images Dataset (AID)...")
    train_loader, val_loader, num_classes = get_dataloaders(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        seed=config.SEED,
        train_fraction=1.0 
    )
    
    strategies = ["linear_probe", "last_block", "selective", "full"]
    
    # 2. Iterate through the selected backbones
    for model_name in config.SELECTED_MODELS:
        print(f"\n>>> Processing Model: {model_name.upper()} <<<")
        
        model_plot_dir = os.path.join(config.PLOTS_DIR, model_name)
        os.makedirs(model_plot_dir, exist_ok=True)
        
        strategy_histories = {}
        unfrozen_percentages = []
        final_accuracies = []
        
        for strategy in strategies:
            print(f"\n--- Strategy: {strategy.upper()} ---")
            
            # Initialize fresh model [cite: 24-30]
            model = get_model(model_name, num_classes=num_classes, pretrained=True)
            
            # Apply specific unfreezing strategy [cite: 54-58]
            model = setup_finetuning_strategy(model, model_name, strategy=strategy, unfreeze_percent=0.20)
            model = model.to(config.DEVICE)
            
            # Calculate and store the percentage of unfrozen parameters 
            percent_unfrozen = get_unfrozen_percentage(model)
            unfrozen_percentages.append(percent_unfrozen)
            print(f"Percentage of parameters unfrozen: {percent_unfrozen:.2f}%")
            
            # Setup Optimizer (fixed hyperparameters for fair comparison) 
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.Adam(trainable_params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
            criterion = nn.CrossEntropyLoss()
            
            # 3. Train the model (Max 30 Epochs, Track Gradients)
            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=config.DEVICE,
                num_epochs=config.MAX_EPOCHS_FULL,
                track_grads=True # Crucial for Scenario 4.2 
            )
            
            strategy_histories[strategy] = history
            # Record the best validation accuracy achieved
            best_val_acc = max(history['val_acc'])
            final_accuracies.append(best_val_acc)
            
            # 4. Process and Plot Gradient Norms 
            # Average the gradient norms over the final 5 epochs for stability
            final_epochs_grads = history['grad_norms'][-5:]
            avg_grad_norms = {}
            for layer_dict in final_epochs_grads:
                for layer_name, norm in layer_dict.items():
                    avg_grad_norms[layer_name] = avg_grad_norms.get(layer_name, 0.0) + norm
            
            for k in avg_grad_norms:
                avg_grad_norms[k] /= len(final_epochs_grads)
                
            grad_plot_path = os.path.join(model_plot_dir, f"02_finetune_grad_norms_{strategy}.png")
            plot_gradient_norms(avg_grad_norms, model_name, strategy, grad_plot_path)
            
            # Save strategy-specific weights
            torch.save(model.state_dict(), os.path.join(config.MODELS_DIR, f"{model_name}_{strategy}.pth"))
            
        # 5. Generate Scenario 4.2 Summary Plots
        
        # Plot A: Convergence Stability Comparison 
        conv_plot_path = os.path.join(model_plot_dir, f"02_finetune_convergence.png")
        plot_convergence_comparison(strategy_histories, model_name, conv_plot_path)
        print(f"\nSaved convergence comparison to {conv_plot_path}")
        
        # Plot B: Accuracy vs Percentage of Unfrozen Parameters 
        # Sort values so the plot line connects sequentially from lowest to highest % unfrozen
        sorted_indices = sorted(range(len(unfrozen_percentages)), key=lambda k: unfrozen_percentages[k])
        sorted_percentages = [unfrozen_percentages[i] for i in sorted_indices]
        sorted_accuracies = [final_accuracies[i] for i in sorted_indices]
        
        acc_vs_params_path = os.path.join(model_plot_dir, f"02_finetune_acc_vs_params.png")
        plot_finetuning_summary(sorted_percentages, sorted_accuracies, save_path=acc_vs_params_path)
        print(f"Saved Accuracy vs. Unfrozen Parameters plot to {acc_vs_params_path}")
        
        print(f"Completed Fine-Tuning Strategies analysis for {model_name}.\n")

if __name__ == "__main__":
    main()