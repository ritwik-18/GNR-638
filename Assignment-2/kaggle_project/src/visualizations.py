import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ==========================================
# 1. Training & Convergence Plots (Scenarios 4.1 & 4.2)
# ==========================================

def plot_training_curves(history, save_path=None, title_suffix=""):
    """
    Plots training and validation accuracy and loss curves.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Curve
    ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(epochs, history['val_loss'], label='Val Loss', marker='s')
    ax1.set_title(f'Convergence Stability (Loss) {title_suffix}')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy Curve
    ax2.plot(epochs, history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(epochs, history['val_acc'], label='Val Acc', marker='s')
    ax2.set_title(f'Accuracy Curve {title_suffix}')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# ==========================================
# 2. Confusion Matrix (Scenario 4.1)
# ==========================================

def plot_confusion_matrix(y_true, y_pred, num_classes=30, save_path=None):
    """
    Generates and saves a heatmap of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# ==========================================
# 3. Feature Embeddings (Scenarios 4.1 & 4.5)
# ==========================================

def plot_embeddings(features, labels, method='PCA', save_path=None, title="Feature Embeddings"):
    """
    Visualizes feature embeddings using PCA or t-SNE.
    """
    if method == 'PCA':
        reducer = PCA(n_components=2)
    elif method == 't-SNE':
        reducer = TSNE(n_components=2, random_state=37)
    else:
        raise ValueError("Method must be 'PCA' or 't-SNE'")
        
    reduced_features = reducer.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_features[:, 0], 
        reduced_features[:, 1], 
        c=labels, 
        cmap='tab20', 
        alpha=0.7, 
        s=15
    )
    plt.colorbar(scatter, label='Class Label')
    plt.title(f'{method} - {title}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# ==========================================
# 4. Fine-Tuning & Layer Probing Summaries (Scenarios 4.2 & 4.5)
# ==========================================

def plot_finetuning_summary(unfrozen_percentages, accuracies, save_path=None):
    """
    Plots validation accuracy vs percentage of unfrozen parameters[cite: 62].
    """
    plt.figure(figsize=(8, 5))
    plt.plot(unfrozen_percentages, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Accuracy vs. Unfrozen Parameters')
    plt.xlabel('Percentage of Unfrozen Parameters (%)')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_layer_probing(depths, accuracies, save_path=None):
    """
    Plots validation accuracy versus network depth for feature probing[cite: 104].
    depths: list of strings (e.g., ['Early', 'Middle', 'Final'])
    """
    plt.figure(figsize=(8, 5))
    plt.plot(depths, accuracies, marker='s', linestyle='-', color='g')
    plt.title('Validation Accuracy vs. Representation Depth')
    plt.xlabel('Network Depth')
    plt.ylabel('Linear Probe Validation Accuracy')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()