import time
import deep_framework as df
from dataset import ImageFolder
from model import MyCNN  # Imports the model definition from model.py

# ==========================================
# 1. Configuration
# ==========================================
# Update these paths to point to your actual extracted dataset folders
TRAIN_DIR = "./Assignments 1 Datasets/Train" 
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.01
SAVE_PATH = "model_final.pkl"

# ==========================================
# [cite_start]2. Complexity Reporting [cite: 47, 101]
# ==========================================
def report_complexity(model):
    print("\n" + "="*40)
    print("      MODEL COMPLEXITY REPORT")
    print("="*40)
    
    total_params = 0
    total_macs = 0
    
    # ---------------------------------------------------------
    # Layer 1: Conv2D (1 -> 6, 5x5)
    # ---------------------------------------------------------
    # Params: (Kernels * In_Channels * H * W) + Biases
    #         (6 * 1 * 5 * 5) + 6 = 156
    # MACs:   (Kernel_Size * In_Channels * Out_H * Out_W) * Out_Channels
    #         Output size is 28x28.
    #         (5 * 5 * 1 * 28 * 28) * 6 = 117,600
    p_c1 = 156
    m_c1 = 117600
    
    # ---------------------------------------------------------
    # Layer 2: Linear (1176 -> 10)
    # ---------------------------------------------------------
    # Params: (In * Out) + Bias
    #         (1176 * 10) + 10 = 11,770
    # MACs:   In * Out
    #         1176 * 10 = 11,760
    p_fc1 = 11770
    m_fc1 = 11760
    
    total_params = p_c1 + p_fc1
    total_macs = m_c1 + m_fc1
    
    # FLOPs are typically approximated as 2 * MACs (multiply + accumulate)
    total_flops = 2 * total_macs
    
    print(f"Model Architecture: Conv(5x5)->ReLU->Pool(2x2)->FC")
    print(f"Total Trainable Parameters: {total_params}")
    print(f"Total MACs per Inference:   {total_macs}")
    print(f"Total FLOPs per Inference:  {total_flops}")
    print("="*40 + "\n")

# ==========================================
# [cite_start]3. Training Loop [cite: 25]
# ==========================================
def main():
    # --- Initialize Model ---
    print("Initializing Network...")
    model = MyCNN()
    optimizer = df.SGD(model.parameters(), LR)
    
    # Report Complexity Requirements
    report_complexity(model)

    # --- Load Dataset ---
    print(f"Loading Dataset from {TRAIN_DIR}...")
    try:
        # [cite_start]The ImageFolder class handles OpenCV loading [cite: 34, 85]
        train_data = ImageFolder(TRAIN_DIR, resize_shape=(32, 32))
        
        # [cite_start]Report Loading Time [cite: 39]
        print(f"Dataset Loading Time: {train_data.load_time:.4f} seconds") 
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Please ensure '{TRAIN_DIR}' exists and contains class folders.")
        return

    print(f"Starting Training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        train_data.shuffle()
        
        total_loss = 0
        correct = 0
        total_samples = 0
        
        num_batches = len(train_data) // BATCH_SIZE
        
        for i in range(num_batches):
            # 1. Get Batch
            x, y = train_data.get_batch(i * BATCH_SIZE, BATCH_SIZE)
            
            # 2. Forward Pass
            logits = model.forward(x)
            
            # 3. Compute Loss
            loss = df.cross_entropy_loss(logits, y)
            
            # [cite_start]4. Backward Pass (Autograd) [cite: 15, 25]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 5. Metrics
            total_loss += loss.data[0]
            
            # Calculate Batch Accuracy
            # (Manually find argmax since we don't have NumPy)
            for b in range(logits.shape[0]):
                # Extract the row of 10 class scores
                row = logits.data[b*10 : (b+1)*10]
                # Find index of max value
                pred = row.index(max(row))
                if pred == int(y.data[b]):
                    correct += 1
            
            total_samples += BATCH_SIZE
            
            # Log every 10 batches
            if i % 10 == 0:
                print(f"  [Epoch {epoch}][Batch {i}/{num_batches}] Loss: {loss.data[0]:.4f}")

        # Epoch Summary
        avg_loss = total_loss / num_batches
        acc = correct / total_samples
        epoch_time = time.time() - start_time
        
        print(f"==== Epoch {epoch} Finished | Time: {epoch_time:.2f}s | Avg Loss: {avg_loss:.4f} | Acc: {acc*100:.2f}% ====")

    # --- Save Model ---
    # [cite_start]Strictly required for the evaluation script to work [cite: 96]
    print(f"\nSaving model weights to {SAVE_PATH}...")
    model.save_weights(SAVE_PATH)
    print("Training Complete.")

if __name__ == "__main__":
    main()