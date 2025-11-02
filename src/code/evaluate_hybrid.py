import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import time  # Import time module for ETA

# Import model definitions (Ensure these are in the correct file path)
from model_definitions import CNN, ResNetModel, PatchSelection, DIF, UFD

# Paths and configurations
MODEL_PATH = "D:/AI_Image_Detection_WebApp/src/newly_trained_model"
TEST_DATA_PATH = "D:/AI_Image_Detection_WebApp/new_preprocessed_dataset/test"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
IMG_SIZE = (64, 64)
CLASSIFICATION_REPORT_FILE = "hybrid_classification_report.txt"
PREDICTION_OUTPUT_FILE = "hybrid_model_predictions.txt"
CONFUSION_MATRIX_FILE = "hybrid_confusion_matrix.png"

# Check if the test data path exists
if not os.path.exists(TEST_DATA_PATH):
    raise FileNotFoundError(f"Test data path does not exist: {TEST_DATA_PATH}")

# Transformations for the test dataset
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

# Load the test dataset and DataLoader
test_dataset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = test_dataset.classes
num_classes = len(class_names)
print(f"Classes: {class_names}")

# Function to load the model
def load_model(model_class, file_name):
    model = model_class().to(DEVICE)
    model_file_path = os.path.join(MODEL_PATH, file_name + ".pth")
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model file not found: {model_file_path}")
    model.load_state_dict(torch.load(model_file_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded model: {file_name}")
    return model

# Load models
cnn_model = load_model(CNN, "CNN")
resnet_model = load_model(ResNetModel, "ResNetModel")
patch_model = load_model(PatchSelection, "PatchSelection")
dif_model = load_model(DIF, "DIF")
ufd_model = load_model(UFD, "UFD")

# List of models and weights (use equal weights for now)
models = [cnn_model, resnet_model, patch_model, dif_model, ufd_model]
weights = [1.0, 1.0, 1.0, 1.0, 1.0]

# Ensure that we have loaded the models and the dataset correctly
print(f"Number of test images: {len(test_loader.dataset)}")

# Evaluation function
def evaluate_models(test_loader, models, class_names, weights):
    all_preds = []
    all_labels = []
    output_lines = []

    # Start the timer to calculate ETA
    start_time = time.time()
    total_batches = len(test_loader)

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            print(f"Processing batch {batch_idx + 1}/{total_batches} with {len(images)} images...")

            # Record time for each batch
            batch_start_time = time.time()

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            votes = []

            # Collect predictions from all models
            for model in models:
                outputs = model(images)
                preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
                votes.append(preds.cpu().numpy())

            # Convert votes to a numpy array
            votes = np.array(votes)  # Shape: [num_models, batch_size]
            weighted_votes = np.zeros((votes.shape[1], len(class_names)))  # [batch_size, num_classes]

            # Apply weighted voting
            for i in range(len(models)):
                for j in range(votes.shape[1]):
                    weighted_votes[j][votes[i][j]] += weights[i]

            # Get final predictions
            final_preds = np.argmax(weighted_votes, axis=1)

            # Store predictions and ground truth labels
            all_preds.extend(final_preds)
            all_labels.extend(labels.cpu().numpy())

            # Save the prediction results with filenames
            for i in range(len(final_preds)):
                img_idx = len(all_preds) - len(final_preds) + i
                img_path = test_dataset.samples[img_idx][0]  # Get image file path
                true_class = class_names[labels[i]]
                predicted_class = class_names[final_preds[i]]
                output_lines.append(f"{img_path} | Actual: {true_class} | Predicted: {predicted_class}\n")

            # Calculate ETA for the current batch
            batch_time = time.time() - batch_start_time
            batches_remaining = total_batches - (batch_idx + 1)
            eta = batch_time * batches_remaining
            eta_minutes = int(eta // 60)
            eta_seconds = int(eta % 60)

            # Print ETA
            print(f"ETA: {eta_minutes}m {eta_seconds}s")

    # Save the predictions to a text file
    with open(PREDICTION_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.writelines(output_lines)
    print(f"Predictions saved to {PREDICTION_OUTPUT_FILE}")

    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("Classification Report:\n", report)

    # Save classification report to a text file
    with open(CLASSIFICATION_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Classification report saved to {CLASSIFICATION_REPORT_FILE}")

    # Generate and save confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE)
    plt.show()
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_FILE}")

# Run the evaluation
evaluate_models(test_loader, models, class_names, weights)
print("Evaluation completed.")
print("All models evaluated successfully.")
print("Hybrid model evaluation completed successfully.")        