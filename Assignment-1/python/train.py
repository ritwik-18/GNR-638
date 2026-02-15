import time
import deep_framework as df 
import sys
from dataset import ImageFolder
from model import MyCNN

# ==========================================
# 1. Configuration
# ==========================================
if len(sys.argv) > 1:
    TRAIN_DIR = sys.argv[1]
else:
    TRAIN_DIR = "./Assignments 1 Datasets/Train"

BATCH_SIZE = 64
EPOCHS = 100
LR = 0.05
SAVE_PATH = "model_final.pkl"


# ==========================================
# 2. Complexity Reporting (Dynamic)
# ==========================================
def report_complexity(num_classes):

    # Conv1
    p_c1 = (16 * 3 * 5 * 5) + 16
    m_c1 = (5 * 5 * 3 * 28 * 28) * 16

    # Conv2
    p_c2 = (32 * 16 * 3 * 3) + 32
    m_c2 = (3 * 3 * 16 * 12 * 12) * 32

    # FC
    p_fc1 = (1152 * num_classes) + num_classes
    m_fc1 = 1152 * num_classes

    total_params = p_c1 + p_c2 + p_fc1
    total_macs = m_c1 + m_c2 + m_fc1
    total_flops = 2 * total_macs

    print("\n" + "="*40)
    print("      MODEL COMPLEXITY REPORT")
    print("="*40)
    print("Architecture: Conv→Pool→Conv→Pool→FC")
    print(f"Total Parameters: {total_params}")
    print(f"Total MACs: {total_macs}")
    print(f"Total FLOPs: {total_flops}")
    print("="*40 + "\n")

# ==========================================
# 3. Training Loop
# ==========================================
def main():

    # --------------------------------------
    # Load Dataset
    # --------------------------------------
    print(f"Loading Dataset from {TRAIN_DIR}...")
    try:
        train_data = ImageFolder(TRAIN_DIR, resize_shape=(32, 32))
        num_classes = len(train_data.class_to_idx)
        print(f"Dataset Loading Time: {train_data.load_time:.4f} seconds")
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Please ensure '{TRAIN_DIR}' exists and contains class folders.")
        return

    # --------------------------------------
    # Initialize Model
    # --------------------------------------
    print("Initializing Network...")
    model = MyCNN(num_classes)
    optimizer = df.SGD(model.parameters(), LR)

    # Report Complexity AFTER knowing num_classes
    report_complexity(num_classes)

    print(f"Starting Training for {EPOCHS} epochs...")

    # --------------------------------------
    # Training
    # --------------------------------------
    for epoch in range(EPOCHS):
        start_time = time.time()

        train_data.shuffle()

        total_loss = 0
        correct = 0
        total_samples = 0

        num_batches = (len(train_data) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in range(num_batches):

            # 1️⃣ Get Batch
            x, y = train_data.get_batch(i * BATCH_SIZE, BATCH_SIZE)

            # 2️⃣ Forward
            logits = model.forward(x)

            # 3️⃣ Loss
            loss = df.cross_entropy_loss(logits, y)

            # 4️⃣ Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data[0]

            # 5️⃣ Accuracy (dynamic classes)
            batch_size = int(y.shape[0])

            for b in range(batch_size):
                row = list(
                    logits.data[b*num_classes : (b+1)*num_classes]
                )
                pred = row.index(max(row))

                if pred == int(y.data[b]):
                    correct += 1

            total_samples += logits.shape[0]

            if i % 10 == 0:
                print(
                    f"  [Epoch {epoch}][Batch {i}/{num_batches}] "
                    f"Loss: {loss.data[0]:.4f}"
                )

        # Epoch summary
        avg_loss = total_loss / num_batches
        acc = correct / total_samples
        epoch_time = time.time() - start_time

        print(
            f"==== Epoch {epoch} Finished | "
            f"Time: {epoch_time:.2f}s | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Acc: {acc*100:.2f}% ===="
        )

    # --------------------------------------
    # Save Model
    # --------------------------------------
    print(f"\nSaving model weights to {SAVE_PATH}...")
    model.save_weights(SAVE_PATH)
    print("Model weights saved.")
    print("Training Complete.")


if __name__ == "__main__":
    main()
