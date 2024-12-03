import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, classification_report, confusion_matrix, roc_auc_score,precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler





class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss()(logits, targets)
        pt = torch.exp(-ce_loss)  # Probability of the correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss


class RecallLoss(nn.Module):
    def __init__(self):
        super(RecallLoss, self).__init__()

    def forward(self, logits, targets):
        preds = torch.sigmoid(logits[:, 1])  # Positive class logits
        labels = targets.float()

        # False Negatives = (1 - predicted) * true positive
        fn_loss = ((1 - preds) * labels).mean()
        return fn_loss


class WeightedPRAUCLoss(nn.Module):
    def __init__(self, positive_weight=1.0):
        super(WeightedPRAUCLoss, self).__init__()
        self.positive_weight = positive_weight

    def forward(self, logits, targets):
        preds = torch.sigmoid(logits[:, 1])  
        labels = targets.float()

        # Pairwise indices
        positive_indices = (labels == 1).nonzero(as_tuple=True)[0]
        negative_indices = (labels == 0).nonzero(as_tuple=True)[0]

        if len(positive_indices) == 0 or len(negative_indices) == 0:
            return torch.tensor(0.0, requires_grad=True)

        positive_preds = preds[positive_indices]
        negative_preds = preds[negative_indices]

        # Pairwise differences
        pairwise_diff = positive_preds.unsqueeze(1) - negative_preds.unsqueeze(0)
        hinge_loss = torch.clamp(1 - pairwise_diff, min=0)

        # Apply weights to minority-class pairs
        weights = torch.ones_like(hinge_loss)
        weights[:, :] = self.positive_weight  

        loss = (weights * hinge_loss).mean()
        return loss
