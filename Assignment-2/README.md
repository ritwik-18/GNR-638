# GNR 638 Assignment 2: Deep Learning on Aerial Imagery

**Team Members:**
* Yasasvi Naidu (23B1037)
* MVV Harsha Vardhan (23B0975)
* B Ritwik (23B0954)

## Project Overview
This repository contains the codebase, raw logs, and the final analysis report for evaluating the transferability of pre-trained visual representations on the Aerial Images Dataset (AID). We benchmarked three distinct architectures—**ResNet50**, **EfficientNet-B0**, and **ConvNeXt-Tiny**—across five experimental scenarios:
1. Linear Probing
2. Fine-Tuning Dynamics
3. Few-Shot Learning Analysis
4. Corruption Robustness
5. Layer-Wise Feature Probing

---

## Important Note on Reproducibility
Due to GitHub's strict 100MB file size limit, the raw AID image dataset and our heavy trained PyTorch model weights (`.pth` files) **have not been uploaded** to this repository. 

If you wish to run the code and reproduce our exact results, please ensure you have the dataset and model weights locally, and place them in the following directories before executing the scripts:
* **Dataset:** Extract the AID dataset into `kaggle_project/train_data/`
* **Model Weights:** Place the pre-trained/fine-tuned `.pth` files into `kaggle_project/outputs/saved_models/`

---

## How to Run the Code

**1. Install Dependencies** Navigate into the main code directory and install the required Python packages. (A GPU-enabled environment is highly recommended).
```bash
cd kaggle_project
pip install -r requirements.txt

2. Execute the Experimental Scenarios Once the data and weights are in place, you can run any of the 5 scenarios using the provided execution scripts. Make sure you are inside the kaggle_project directory when running these commands:

# Scenario 1: Linear Probing Baseline
python scripts/run_1_linear_probe.py

# Scenario 2: Fine-Tuning Strategies
python scripts/run_2_finetuning.py

# Scenario 3: Data Efficiency (Few-Shot)
python scripts/run_3_few_shot.py

# Scenario 4: Corruption Robustness
python scripts/run_4_robustness.py

# Scenario 5: Layer-Wise Semantic Transferability
python scripts/run_5_layer_probing.py
