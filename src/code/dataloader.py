# dataloader.py
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_test_loader(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, test_dataset.classes
