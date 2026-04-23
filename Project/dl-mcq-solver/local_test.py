import os
import time
from src.config import CFG
from src.sandbox import Sandbox
from src.solver import load_model, solve_one
from src.utils import TimeBudget

# 1. Initialize
print("Starting Local Test...")
model, processor = load_model()
sandbox = Sandbox(timeout=CFG.SANDBOX_TIMEOUT)

sample_dir = "./sample_test_cases"
image_files = [f for f in os.listdir(sample_dir) if f.endswith(".png")]

# 2. Simulate the Kaggle loop
budget = TimeBudget(total_seconds=300, n_questions=len(image_files))

for img_file in image_files:
    image_path = os.path.join(sample_dir, img_file)
    deadline = budget.next_deadline()
    
    print(f"\n--- Testing: {img_file} ---")
    result = solve_one(
        model=model,
        processor=processor,
        image_path=image_path,
        sandbox=sandbox,
        deadline=deadline
    )
    
    budget.mark_done()
    sandbox.reset()

print("\nLocal testing complete. Check the logs above for routing and answers.")
sandbox.close()