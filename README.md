# Robust Adversarial Training 🛡️ 

This project implements a robust image classifier capable of resisting adversarial attacks using Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD). We utilize ResNet-50 for classification and apply adversarial training to enhance the model's resilience on both clean and perturbed data.

## Project Overview
1. Objective :- Train a classifier that maintains high accuracy on both clean images and adversarially perturbed images.
2. Model Used :- Model: ResNet-50 (pre-trained on ImageNet)
3. Adverserial Techniques :- FGSM and PGD
4. Training Enhancements: Adversarial sample generation, stratified data splitting, RGB normalization, checkpointing.

##Folder Structure

.
├── train/
│   └── models_variants       # set of models trained while experimentation
│   └── model_training.py       # Main script for training and adversarial defense
│   └── final_model.pt       # best case model
├── eda/
│   └── preprocessing.ipynb
│   └──train_test_class_balance.ipynb
.submission.py

## Adversarial Attacks Implemented
1. FGSM (Fast Gradient Sign Method)
 Perturbation: epsilon = 0.01 with Single-step attack using loss gradients.

2. PGD (Projected Gradient Descent)
  Perturbation: epsilon = 0.03, alpha = 0.007, iters = 10. Iterative and stronger than FGSM.

50% of batches during training are adversarial, randomly alternating between FGSM and PGD.

## Highlights
1. Stratified Split: Ensures fair distribution of rare labels (with <2 samples).
2. Grayscale to RGB Conversion: Uniform input size of 3×32×32 for ResNet.
3. Checkpoints: Epoch-wise model saving, especially from epoch 10 onward.
4. Elbow Curve Detection: Used for early stopping and model selection to avoid overfitting.

## Results
Best Model: epoch_17 (highest test accuracy)

## Challenges Addressed
1. Trade-off between robustness and clean accuracy.
2. Avoiding overfitting while maintaining adversarial generalization.
3. Efficient adversarial training without large computational overhead.
