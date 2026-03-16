import os
import sys
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.dataset import get_dataloaders, get_layer_probing_loader
from src.models import FeatureExtractor
from src.visualizations import plot_layer_probing, plot_embeddings

def extract_features(extractor, dataloader, device):
    """
    Passes a dataset through the FeatureExtractor and concatenates the 
    early, middle, and final representations into numpy arrays.
    """
    extractor.eval()
    
    features_dict = {"early": [], "middle": [], "final": []}
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            # Extractor returns a dictionary of pooled features
            out = extractor(inputs)
            
            features_dict["early"].append(out["early"].cpu().numpy())
            features_dict["middle"].append(out["middle"].cpu().numpy())
            features_dict["final"].append(out["final"].cpu().numpy())
            
            all_labels.extend(labels.numpy())
            
    # Concatenate list of batch arrays into single large arrays
    for key in features_dict:
        features_dict[key] = np.vstack(features_dict[key])
        
    return features_dict, np.array(all_labels)

def calculate_feature_norms(features_matrix):
    """Calculates the average L2 norm of the feature vectors."""
    # features_matrix shape: (num_samples, feature_dim)
    norms = np.linalg.norm(features_matrix, axis=1)
    return np.mean(norms), np.std(norms)

def main():
    print("="*60)
    print("Starting Scenario 4.5: Layer-Wise Feature Probing")
    print("="*60)
    
    # 1. Load Data
    # For training the linear probes, we use the standard train/val split
    print("Loading standard Train/Val splits for probe training...")
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE, seed=config.SEED
    )
    
    # For the PCA plots, we MUST use the fixed 30 samples/class subset 
    print(f"Loading fixed PCA probing subset (Seed: {config.SEED})...")
    pca_loader = get_layer_probing_loader(
        data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE, seed=config.SEED
    )
    
    for model_name in config.SELECTED_MODELS:
        print(f"\n>>> Analyzing Representations for: {model_name.upper()} <<<")
        
        model_plot_dir = os.path.join(config.PLOTS_DIR, model_name)
        os.makedirs(model_plot_dir, exist_ok=True)
        
        # Initialize Feature Extractor (Backbone is strictly frozen) 
        extractor = FeatureExtractor(model_name, pretrained=True).to(config.DEVICE)
        
        # 2. Extract Features for the Train and Val sets
        print("Extracting features for the training set (this may take a minute)...")
        train_feats, train_labels = extract_features(extractor, train_loader, config.DEVICE)
        
        print("Extracting features for the validation set...")
        val_feats, val_labels = extract_features(extractor, val_loader, config.DEVICE)
        
        depths = ["early", "middle", "final"]
        accuracies = []
        norm_stats = {}
        
        # 3. Train Linear Classifiers (Probes) per depth 
        for depth in depths:
            print(f"\n--- Training Probe for {depth.upper()} Layer ---")
            X_train, y_train = train_feats[depth], train_labels
            X_val, y_val = val_feats[depth], val_labels
            
            # Record feature norm statistics [cite: 105]
            mean_norm, std_norm = calculate_feature_norms(X_val)
            norm_stats[depth] = (mean_norm, std_norm)
            print(f"Feature Norm (Val): Mean = {mean_norm:.4f}, Std = {std_norm:.4f}")
            
            # Standardize features before applying Logistic Regression
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train Linear Probe
            clf = LogisticRegression(max_iter=1000, n_jobs=-1)
            clf.fit(X_train_scaled, y_train)
            
            # Evaluate
            preds = clf.predict(X_val_scaled)
            acc = accuracy_score(y_val, preds)
            accuracies.append(acc)
            print(f"{depth.capitalize()} Layer Validation Accuracy: {acc:.4f}")
            
        # 4. Generate Accuracy vs Depth Plot [cite: 104]
        acc_plot_path = os.path.join(model_plot_dir, f"05_layer_probing_accuracy.png")
        plot_layer_probing([d.capitalize() for d in depths], accuracies, save_path=acc_plot_path)
        print(f"\nSaved Accuracy vs Depth plot to {acc_plot_path}")
        
        # 5. Extract Features for the Fixed PCA Subset and Plot 
        print("\nExtracting representations for the fixed PCA subset...")
        pca_feats, pca_labels = extract_features(extractor, pca_loader, config.DEVICE)
        
        for depth in depths:
            pca_path = os.path.join(model_plot_dir, f"05_layer_probing_pca_{depth}.png")
            plot_embeddings(
                features=pca_feats[depth], 
                labels=pca_labels, 
                method='PCA', 
                save_path=pca_path, 
                title=f"{model_name.upper()} - {depth.capitalize()} Layer PCA"
            )
        print(f"Saved PCA plots for early, middle, and final layers.")
        
        print(f"Completed Layer-Wise Feature Probing for {model_name}.\n")

if __name__ == "__main__":
    main()