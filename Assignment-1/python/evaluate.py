import sys
import os
import deep_framework as df
from dataset import ImageFolder
from model import MyCNN

def evaluate(test_dir, weight_path):
    # 1. Load Model structure
    print(f"Initializing model...")
    model = MyCNN()
    
    # 2. Load Trained Weights
    try:
        model.load_weights(weight_path)
    except FileNotFoundError:
        print(f"Error: Weight file not found at {weight_path}")
        return

    # 3. Load Test Data
    print(f"Loading test data from {test_dir}...")
    # Resize to 32x32 as required [cite: 35]
    test_data = ImageFolder(test_dir, resize_shape=(32, 32))
    
    total = len(test_data)
    if total == 0:
        print("No images found in test directory.")
        return

    correct = 0
    BATCH_SIZE = 64
    
    print("Starting Evaluation...")
    
    # 4. Evaluation Loop
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(num_batches):
        x, y = test_data.get_batch(i * BATCH_SIZE, BATCH_SIZE)
        
        # Forward Pass
        logits = model.forward(x)
        
        # Calculate Accuracy
        # Manual Argmax since NumPy is prohibited [cite: 86]
        for b in range(logits.shape[0]):
            # Get the 10 class scores for this image
            row = logits.data[b*10 : (b+1)*10]
            # Find index of max score
            pred = row.index(max(row))
            
            if pred == int(y.data[b]):
                correct += 1
                
    acc = correct / total
    print(f"Dataset Loading Time: {test_data.load_time:.4f}s")
    print(f"Test Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    # The evaluation script must run using only parent directory path and weights path [cite: 96]
    if len(sys.argv) < 3:
        print("Usage: python3 evaluate.py <test_dir_path> <weight_path>")
        # Fallback defaults for local testing
        test_dir = "./Assignments 1 Datasets/Test"
        weight_path = "model_final.pkl"
    else:
        test_dir = sys.argv[1]
        weight_path = sys.argv[2]
        
    evaluate(test_dir, weight_path)