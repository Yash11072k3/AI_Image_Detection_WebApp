# Updated training notebook with new dataset paths and all models including ETA
# Dataset paths
raw_dataset_path = r"D:\\AI_Image_Detection_WebApp\\raw dataset"
preprocessed_dataset_path = r"D:\\AI_Image_Detection_WebApp\\new_preprocessed_dataset"

# 1. Import libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import time

# 2. Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Define transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# 4. Load preprocessed dataset
train_dataset = datasets.ImageFolder(os.path.join(preprocessed_dataset_path, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(preprocessed_dataset_path, "val"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(preprocessed_dataset_path, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 5. Define all models
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * 32 * 32, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)

class PatchSelection(nn.Module):
    def __init__(self):
        super(PatchSelection, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 32 * 32, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DIF(nn.Module):
    def __init__(self):
        super(DIF, self).__init__()
        self.fc = nn.Linear(3 * 64 * 64, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

class UFD(nn.Module):
    def __init__(self):
        super(UFD, self).__init__()
        self.fc = nn.Linear(3 * 64 * 64, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)
# 6. Function to train a model

from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt

def train_model(model, model_name, resume=False):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 3
    start_epoch = 0

    # Create directory
    base_dir = "newly_trained_model"
    os.makedirs(base_dir, exist_ok=True)
    checkpoint_path = os.path.join(base_dir, f"{model_name}_checkpoint.pth")
    model_save_path = os.path.join(base_dir, f"{model_name}.pth")

    # Resume logic
    if resume and os.path.exists(checkpoint_path):
        print(f"🔄 Resuming {model_name} from checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"👉 Resumed at epoch {start_epoch + 1}")

    print(f"\n🚀 Starting training for {model_name} from epoch {start_epoch + 1}...\n")

    train_accuracies = []
    val_accuracies = []
    losses = []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start_time = time.time()

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training {model_name}", dynamic_ncols=True)
        for batch_idx, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            avg_loss = running_loss / (batch_idx + 1)
            train_acc = 100.0 * correct / total

            elapsed = time.time() - epoch_start_time
            iters_left = len(train_loader) - (batch_idx + 1)
            eta_seconds = int(iters_left / (batch_idx + 1 + 1e-8) * elapsed)
            eta = str(datetime.timedelta(seconds=eta_seconds))

            progress_bar.set_postfix({
                "loss": f"{avg_loss:.3f}",
                "acc": f"{train_acc:.2f}%",
                "eta": eta
            })

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        epoch_time = time.time() - epoch_start_time
        final_loss = running_loss / len(train_loader)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc * 100)
        losses.append(final_loss)

        print(f"✅ {model_name} Epoch {epoch+1} | Loss: {final_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc*100:.2f}% | Time: {epoch_time:.2f}s")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': final_loss
        }, checkpoint_path)

    # Save final model
    torch.save(model.state_dict(), model_save_path)
    print(f"\n🎉 Final model saved at: {model_save_path}")
    print(f"📦 Checkpoint stored at: {checkpoint_path}")
    print(f"✅ {model_name} training completed!\n")

    # Plot Accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(range(start_epoch + 1, num_epochs + 1), train_accuracies, marker='o', label='Train Accuracy')
    plt.plot(range(start_epoch + 1, num_epochs + 1), val_accuracies, marker='x', label='Validation Accuracy')
    plt.title(f"Accuracy per Epoch - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_path = os.path.join(base_dir, f"{model_name}_accuracy.png")
    plt.savefig(acc_path)
    plt.show()

    # Plot Loss
    plt.figure(figsize=(8, 4))
    plt.plot(range(start_epoch + 1, num_epochs + 1), losses, marker='s', color='red')
    plt.title(f"Loss per Epoch - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    loss_path = os.path.join(base_dir, f"{model_name}_loss.png")
    plt.savefig(loss_path)
    plt.show()

# 7. Train all models
train_model(CNN(), "CNN")
train_model(ResNetModel(), "NDD_ResNet")
train_model(PatchSelection(), "PatchSelection")
train_model(DIF(), "DIF")
train_model(UFD(), "UFD")
