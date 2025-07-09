# Robust Adversarial Training 🛡️

This project implements a robust image classifier that resists adversarial attacks by leveraging **Fast Gradient Sign Method (FGSM)** and **Projected Gradient Descent (PGD)**. Using a **ResNet-50** backbone pre-trained on ImageNet, we apply adversarial training to boost the model’s resilience on both clean and adversarially perturbed images.

---

## Project Overview

- **Objective:**  
  Train a classifier that maintains high accuracy on both clean and adversarially perturbed images.

- **Model:**  
  ResNet-50 (pre-trained on ImageNet)

- **Adversarial Techniques:**  
  - FGSM (Fast Gradient Sign Method)  
  - PGD (Projected Gradient Descent)

- **Training Enhancements:**  
  - Adversarial sample generation integrated into training  
  - Stratified data splitting for balanced label distribution  
  - RGB normalization (including grayscale to RGB conversion)  
  - Checkpointing with epoch-wise model saving  
  - Early stopping via elbow curve detection

---

## Folder Structure
.
├── train/
│   ├── models_variants/           # Different model versions from experimentation
│   ├── model_training.py          # Main training script with adversarial defense
│   └── final_model.pt             # Best performing trained model
├── eda/
│   ├── preprocessing.ipynb       # Data preprocessing and cleaning
│   └── train_test_class_balance.ipynb  # Class balance analysis
└── submission.py                 # Submission script


---

## Adversarial Attacks Implemented

1. **FGSM (Fast Gradient Sign Method)**  
   - Perturbation: epsilon = 0.01  
   - Single-step attack using loss gradients

2. **PGD (Projected Gradient Descent)**  
   - Perturbation: epsilon = 0.03, alpha = 0.007, iterations = 10  
   - Iterative and stronger attack compared to FGSM

**Training Detail:**  
50% of training batches contain adversarial samples, alternating randomly between FGSM and PGD attacks.

---

## Highlights

- **Stratified Split:**  
  Ensures fair distribution of rare labels (those with fewer than 2 samples).

- **Grayscale to RGB Conversion:**  
  Standardizes input images to 3×32×32 channels suitable for ResNet.

- **Checkpointing:**  
  Saves model weights after each epoch, with special focus on epochs ≥ 10.

- **Elbow Curve Detection:**  
  Used for early stopping and selecting the best model to avoid overfitting.

---

## Results

- **Best Model:** `epoch_17` with highest test accuracy.

---

## Challenges Addressed

- Balancing robustness to adversarial examples with clean image accuracy.
- Preventing overfitting while maintaining adversarial generalization.
- Efficient adversarial training without excessive computational cost.
