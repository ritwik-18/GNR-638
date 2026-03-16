import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add the parent directory to the system path so we can import our src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.dataset import get_dataloaders
from src.models import get_model, setup_finetuning_strategy
from src.utils import calculate_efficiency_metrics, train_model, evaluate
from src.visualizations import plot_training_curves, plot_confusion_matrix, plot_embeddings

def extract_embeddings(model, dataloader, device):
    """
    Helper function to extract intermediate features (before the linear head)
    for PCA/t-SNE visualization[cite: 50].
    """
    model.eval()
    all_features = []
    all_labels = []
    
    # Universal pooling for CNN feature maps
    pool = nn.AdaptiveAvgPool2d(1)
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            # Use timm's forward_features to bypass the classification head
            features = model.forward_features(inputs)
            # Pool spatial dimensions and flatten
            features = pool(features).flatten(1)
            
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return np.vstack(all_features), np.array(all_labels)

def main():
    print("="*50)
    print("Starting Scenario 4.1: Linear Probe Transfer")
    print("="*50)
    
    # 1. Load Dataset (100% of training data)
    print("Loading Aerial Images Dataset (AID)...")
    train_loader, val_loader, num_classes = get_dataloaders(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        seed=config.SEED,
        train_fraction=1.0 # Use full dataset
    )
    
    # 2. Iterate through the selected backbones
    for model_name in config.SELECTED_MODELS:
        print(f"\n>>> Processing Model: {model_name.upper()} <<<")
        
        # Initialize model [cite: 24-30]
        model = get_model(model_name, num_classes=num_classes, pretrained=True)
        
        # Apply Linear Probe strategy (freeze backbone, train head) [cite: 43-46]
        model = setup_finetuning_strategy(model, model_name, strategy="linear_probe")
        model = model.to(config.DEVICE)
        
        # Print Efficiency Metrics 
        calculate_efficiency_metrics(model, device=config.DEVICE)
        
        # Setup Optimizer: ONLY pass parameters that require gradients
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()
        
        # 3. Train the model (Max 30 Epochs) 
        print(f"Training linear classifier for {model_name}...")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=config.DEVICE,
            num_epochs=config.MAX_EPOCHS_FULL,
            track_grads=False # Not needed for 4.1
        )
        
        # 4. Generate & Save Visualizations
        model_plot_dir = os.path.join(config.PLOTS_DIR, model_name)
        os.makedirs(model_plot_dir, exist_ok=True)
        
        # Plot A: Training and Validation Curves [cite: 48]
        curve_path = os.path.join(model_plot_dir, f"01_linear_probe_curves.png")
        plot_training_curves(history, save_path=curve_path, title_suffix=f"({model_name} - Linear Probe)")
        print(f"Saved training curves to {curve_path}")
        
        # 5. Final Evaluation for Confusion Matrix
        print("Evaluating validation set for confusion matrix...")
        _, _, all_preds, all_targets = evaluate(model, val_loader, criterion, config.DEVICE)
        
        # Plot B: Confusion Matrix [cite: 49]
        cm_path = os.path.join(model_plot_dir, f"01_linear_probe_cm.png")
        plot_confusion_matrix(all_targets, all_preds, num_classes=num_classes, save_path=cm_path)
        print(f"Saved confusion matrix to {cm_path}")
        
        # Plot C: Feature Embeddings Visualization (PCA) [cite: 50]
        print("Extracting features for PCA embedding visualization...")
        features, labels = extract_embeddings(model, val_loader, config.DEVICE)
        
        pca_path = os.path.join(model_plot_dir, f"01_linear_probe_pca.png")
        plot_embeddings(features, labels, method='PCA', save_path=pca_path, title=f"{model_name} Validation Embeddings")
        print(f"Saved PCA embeddings to {pca_path}")
        
        # Save the model weights
        torch.save(model.state_dict(), os.path.join(config.MODELS_DIR, f"{model_name}_linear_probe.pth"))
        print(f"Completed linear probe for {model_name}.\n")

if __name__ == "__main__":
    main()