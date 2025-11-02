import torch
import torch.nn as nn
from torchvision.models import resnet18



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
