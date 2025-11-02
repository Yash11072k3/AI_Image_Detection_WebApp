
# utils.py

import os   
import numpy as np
import torch
from torch import nn
from typing import List, Type

def load_model(model_class, file_name, model_path, device):
    model = model_class().to(device)
    model_file_path = os.path.join(model_path, file_name + ".pth")
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model file not found: {model_file_path}")
    model.load_state_dict(torch.load(model_file_path, map_location=device))
    model.eval()
    print(f"Loaded model: {file_name}")
    return model

def weighted_voting(preds, weights, num_classes):
    vote_count = np.zeros(num_classes)
    for i, pred in enumerate(preds):
        vote_count[pred] += weights[i]
    return int(np.argmax(vote_count))