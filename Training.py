import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os

# =============================
# 1. Configuration
# =============================
data_dir = "training_dataset"   # must contain class subfolders
num_classes = 2                 # POS vs Phone
num_epochs = 10
batch_size = 8
learning_rate = 0.001
k_folds = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# =============================
# 2. Transforms
# =============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# =============================
# 3. Dataset
# =============================
dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
class_names = dataset.classes
print("Classes:", class_names)

# =============================
# 4. Loss function with weights
# =============================
# Higher weight for POS class to reduce false negatives
class_counts = np.bincount(dataset.targets)
weights = torch.tensor([1.0, 3.0], dtype=torch.float).to(device)  # assume [phone, pos]
criterion = nn.CrossEntropyLoss(weight=weights)

# =============================
# 5. K-Fold Training
# =============================
kfold = KFold(n_splits=k_folds, shuffle=True)

for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    print(f"\nFOLD {fold+1}")
    print("-" * 20)

    # Subset samplers
    train_subset = Subset(dataset, train_ids)
    val_subset = Subset(dataset, val_ids)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Model
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    best_model_wts = None

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_subset)

        # Validation
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Acc: {acc:.4f}")

        # Save best for this fold
        if acc > best_acc:
            best_acc = acc
            best_model_wts = model.state_dict()

    # Save best model for this fold
    fold_model_path = os.path.join(save_dir, f"mobilenet_pos_vs_phone_fold{fold+1}.pth")
    torch.save(best_model_wts, fold_model_path)
    print(f"Best model for fold {fold+1} saved at {fold_model_path} with acc={best_acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix Fold {fold+1}")
    plt.show()
