import os
import torch

# ==========================================
# 1. Base Paths & Directory Setup
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "train_data") # Assuming dataset is extracted here [cite: 35]
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "saved_models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Create output directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==========================================
# 2. Assignment Constraints & Constants
# ==========================================
SEED = 37  # Group-specific seed for reproducibility 
NUM_CLASSES = 30 # Aerial Images Dataset classes [cite: 33]
MAX_EPOCHS_FULL = 1 # Max epochs for 100% data training [cite: 120]
MAX_EPOCHS_FEW_SHOT = 20 # Max epochs for 5% and 20% data training [cite: 121]

# Scenario 4.5 specific constraint
PCA_SAMPLES_PER_CLASS = 30 # [cite: 106]
TOTAL_PCA_SAMPLES = NUM_CLASSES * PCA_SAMPLES_PER_CLASS

# ==========================================
# 3. Model Selection
# ==========================================
# The assignment requires exactly 3 models[cite: 24]. 
# You can change these to any 3 from the allowed list: 
# ['resnet50', 'inception_v3', 'densenet121', 'efficientnet_b0', 'convnext_tiny']
SELECTED_MODELS = [
    "resnet50",
    "efficientnet_b0",
    "convnext_tiny"
]

# ==========================================
# 4. Global Hyperparameters
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Few-shot splits [cite: 71, 72]
FEW_SHOT_FRACTIONS = {
    "100_percent": 1.0,
    "20_percent": 0.20,
    "5_percent": 0.05
}

# Corruption intensities for Scenario 4.4 [cite: 84]
GAUSSIAN_NOISE_SIGMAS = [0.05, 0.1, 0.2]