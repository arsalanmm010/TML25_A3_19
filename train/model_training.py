import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from tqdm import tqdm
from typing import Tuple
from torchvision import transforms
import torch.nn.functional as F


transform = transforms.Compose([
    transforms.ToTensor(),  # convert PIL image to tensor
])

if __name__ == "__main__":

    # ====== Custom Dataset ======
    class TaskDataset(Dataset):
        def __init__(self, transform=None):
            self.ids = []
            self.imgs = []
            self.labels = []
            self.transform = transform

        def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
            id_ = self.ids[index]
            img = self.imgs[index]
            if not self.transform is None:
                img = self.transform(img)
            label = self.labels[index]
            return id_, img, label

        def __len__(self):
            return len(self.ids)

    def to_rgb(image):
        return image.convert('RGB') if image.mode != 'RGB' else image

    dataset_data = torch.load("Train.pt", weights_only=False)
    dataset = TaskDataset(transform=transform)
    for id_, img, label in dataset_data:
        dataset.ids.append(int(id_))          # Convert id_ to int here
        dataset.imgs.append(to_rgb(img))
        dataset.labels.append(int(label))

    # ====== Stratified Split with Rare Tag Handling ======
    labels_np = np.array(dataset.labels)
    tag_counts = Counter(labels_np)
    rare_indices = np.array([i for i, tag in enumerate(labels_np) if tag_counts[tag] < 2])
    strat_eligible = np.array([i for i, tag in enumerate(labels_np) if tag_counts[tag] >= 2])
    strat_tags = labels_np[strat_eligible]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx_r, test_idx_r = next(sss.split(strat_eligible, strat_tags))

    train_idx_r = strat_eligible[train_idx_r]
    test_idx_r = strat_eligible[test_idx_r]
    train_idx = np.concatenate([rare_indices, train_idx_r])
    test_idx = test_idx_r

    train_idx = train_idx.astype(int)
    test_idx = test_idx.astype(int)

    train_data = Subset(dataset, train_idx)
    test_data = Subset(dataset, test_idx)

    train_loader = DataLoader(train_data, batch_size=512, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False, num_workers=0)

    # ====== Model (ResNet50) ======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(set(dataset.labels)))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ====== Training & Evaluation Functions ======
    # --- FGSM Attack ---
    def fgsm_sample_training(model, images, labels, epsilon=0.01, targeted=False):
        images = images.clone().detach().requires_grad_(True)
        outputs = model(images)

        loss = F.cross_entropy(outputs, labels)
        if targeted:
            loss = -loss  # Minimize loss for targeted attack

        model.zero_grad()
        loss.backward()

        grad_sign = images.grad.data.sign()
        perturbed_images = images + epsilon * grad_sign if not targeted else images - epsilon * grad_sign
        perturbed_images = torch.clamp(perturbed_images, 0, 1)

        return perturbed_images.detach()

    # --- PGD Attack ---
    def pgd_sample_training(model, images, labels, eps=0.03, alpha=0.007, iters=10, random_start=True, targeted=False):
        ori_images = images.clone().detach()

        # Random start within the epsilon ball
        if random_start:
            perturbed_images = ori_images + torch.empty_like(ori_images).uniform_(-eps, eps)
            perturbed_images = torch.clamp(perturbed_images, 0, 1)
        else:
            perturbed_images = ori_images.clone().detach()

        for _ in range(iters):
            perturbed_images.requires_grad = True
            outputs = model(perturbed_images)

            loss = F.cross_entropy(outputs, labels)
            if targeted:
                loss = -loss

            model.zero_grad()
            loss.backward()

            grad_sign = perturbed_images.grad.data.sign()
            step = alpha * grad_sign if not targeted else -alpha * grad_sign

            adv_images = perturbed_images + step
            eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
            perturbed_images = torch.clamp(ori_images + eta, 0, 1).detach()

        return perturbed_images

    # --- Training Function ---
    def train(model, loader, use_adv=True):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for _, inputs, targets in tqdm(loader, desc="Training"):
            inputs, targets = inputs.to(device), targets.to(device)

            # Use adversarial examples only for half of the batches
            if use_adv and torch.rand(1).item() < 0.5:
                # Choose either FGSM or PGD randomly for diversity
                if torch.rand(1).item() < 0.5:
                    inputs = fgsm_sample_training(model, inputs, targets, epsilon=0.01)
                else:
                    inputs = pgd_sample_training(model, inputs, targets, eps=0.03, alpha=0.007, iters=5)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        return running_loss / len(loader), 100 * correct / total

    def evaluate(model, loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _, inputs, targets in tqdm(loader, desc="Evaluating"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        return 100 * correct / total

    # ====== Run Training ======
    epochs = 20
    for epoch in range(epochs):
        loss, train_acc = train(model, train_loader)
        test_acc = evaluate(model, test_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        if epoch >= 10: 
            # Save model after each epoch
            torch.save(model.state_dict(), f"resnet50_epoch{epoch+1}.pt")
            print(f"💾 Model saved to resnet50_epoch{epoch+1}.pt")
