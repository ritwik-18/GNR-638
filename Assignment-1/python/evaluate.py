import sys
import deep_framework as df
from dataset import ImageFolder
from model import MyCNN


def evaluate(test_dir, weight_path):

    # --------------------------------------
    # 1️⃣ Load Test Data FIRST
    # --------------------------------------
    print(f"Loading test data from {test_dir}...")
    try:
        test_data = ImageFolder(test_dir, resize_shape=(32, 32))
        num_classes = len(test_data.class_to_idx)
        print(f"Dataset Loading Time: {test_data.load_time:.4f}s")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    if len(test_data) == 0:
        print("No images found in test directory.")
        return

    # --------------------------------------
    # 2️⃣ Initialize Model with correct classes
    # --------------------------------------
    print("Initializing model...")
    model = MyCNN(num_classes)

    # --------------------------------------
    # 3️⃣ Load Weights
    # --------------------------------------
    try:
        model.load_weights(weight_path)
        print(f"Model weights loaded from {weight_path}")
    except FileNotFoundError:
        print(f"Error: Weight file not found at {weight_path}")
        return

    # --------------------------------------
    # 4️⃣ Evaluation
    # --------------------------------------
    correct = 0
    total = len(test_data)
    BATCH_SIZE = 64

    print("Starting Evaluation...")

    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(num_batches):
        x, y = test_data.get_batch(i * BATCH_SIZE, BATCH_SIZE)

        logits = model.forward(x)

        for b in range(logits.shape[0]):
            row = list(
                logits.data[b*num_classes : (b+1)*num_classes]
            )
            pred = row.index(max(row))

            if pred == int(y.data[b]):
                correct += 1

    acc = correct / total
    print(f"Test Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python3 evaluate.py <test_dir_path> <weight_path>")
        sys.exit(1)

    test_dir = sys.argv[1]
    weight_path = sys.argv[2]

    evaluate(test_dir, weight_path)
