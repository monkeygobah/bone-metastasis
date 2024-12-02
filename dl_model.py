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

class MLP(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 2) 
        )

    def forward(self, x):
        return self.model(x)

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


def get_balanced_dataloader(X_train_tensor, y_train_tensor, batch_size):
    # Calculate class weights
    class_counts = torch.bincount(y_train_tensor.long())
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train_tensor.long()]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    balanced_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    return balanced_loader


def get_dataloaders(X_train_bone, X_test_bone, y_train_bone, y_test_bone, batch_size=32):
    scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train_bone)
    # X_test_scaled = scaler.transform(X_test_bone)

    X_train_scaled = X_train_bone.to_numpy() 
    X_test_scaled = X_test_bone.to_numpy()

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_bone.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_bone.values, dtype=torch.float32)

    positive_samples = y_train_tensor.sum().item()
    negative_samples = len(y_train_tensor) - positive_samples
    pos_weight = negative_samples / positive_samples
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    balanced_loader = get_balanced_dataloader(X_train_tensor, y_train_tensor, batch_size)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    normal_loader = DataLoader(train_dataset, batch_size=batch_size)

    return balanced_loader,normal_loader, test_loader, pos_weight_tensor

import csv
import pandas as pd

def engine(X_train_bone, X_test_bone, y_train_bone, y_test_bone):
    # Get DataLoaders and class weight
    balanced_loader, normal_loader, test_loader, pos_weight_tensor = get_dataloaders(
        X_train_bone, X_test_bone, y_train_bone, y_test_bone, batch_size=32
    )

    # Define model input size
    input_size = X_train_bone.shape[1]

    # Define the loss functions
    loss_functions = {
        "loss_pr": WeightedPRAUCLoss(positive_weight=1),
        "loss_rec": RecallLoss(),
        "loss_cel": nn.CrossEntropyLoss(),
        "loss_foc": FocalLoss(alpha=1, gamma=2),
        "loss_celw": nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight_tensor.item()])),
    }

    # Train a model for each loss function
    for loss_name, loss_fn in loss_functions.items():
        print(f"\nTraining with {loss_name}...\n")
        model = MLP(input_size)
        train_loop(
            model,
            balanced_loader,
            test_loader,
            loss_fn,
            num_epochs=100,
            lr=0.001,
            loss_name=loss_name
        )


def train_loop(model, train_loader, test_loader, loss_fn, num_epochs=100, lr=0.001, loss_name="loss"):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # L2 regularization

    # CSV file to log results
    csv_file = f"{loss_name}_metrics.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Precision", "Recall", "Accuracy", "AUROC", "F1", "Best Epoch"])

    best_metrics = {"Precision": 0, "Recall": 0, "Accuracy": 0, "AUROC": 0, "F1": 0, "Epoch": 0}

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)

            # Calculate loss
            loss = loss_fn(outputs, y_batch.long())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs:
            test_accuracy, precision, recall, auroc, f1 = evaluate_metrics_extended(model, test_loader)

            print(
                f"Epoch {epoch}/{num_epochs} - Train Loss: {epoch_loss / len(train_loader):.4f}, "
                f"Test Accuracy: {test_accuracy:.4f}, Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, AUROC: {auroc:.4f}, F1: {f1:.4f}"
            )

            # Check if this epoch has the best recall
            if recall > best_metrics["F1"]:
                best_metrics.update({"Precision": precision, "Recall": recall, "Accuracy": test_accuracy,
                                     "AUROC": auroc, "F1": f1, "Epoch": epoch})

            # Append metrics to CSV
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, precision, recall, test_accuracy, auroc, f1, best_metrics["Epoch"]])

    print(f"Best metrics for {loss_name}: {best_metrics}")


def evaluate_metrics_extended(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability of the positive class
            predictions = torch.argmax(outputs, axis=1)

            y_true.extend(y_batch.numpy())
            y_pred.extend(predictions.numpy())
            y_probs.extend(probabilities.numpy())

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    auroc = roc_auc_score(y_true, y_probs)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return accuracy, precision, recall, auroc, f1




# def engine(X_train_bone, X_test_bone, y_train_bone, y_test_bone):
#     # Get DataLoaders and class weight
#     # from imblearn.over_sampling import RandomOverSampler

#     # ros = RandomOverSampler(sampling_strategy='minority')
    
#     # X_resampled, y_resampled = ros.fit_resample(X_train_bone, y_train_bone)
    
#     balanced_loader, normal_loader, test_loader, pos_weight_tensor = get_dataloaders(
#         X_train_bone, X_test_bone, y_train_bone, y_test_bone, batch_size=32
#     )

#     # Define model
#     input_size = X_train_bone.shape[1]
#     model = MLP(input_size)

#     # Train the model
#     train_loop(model, balanced_loader, test_loader, pos_weight_tensor, num_epochs=100, lr=0.001)



# def train_loop(model, train_loader, test_loader, pos_weight, num_epochs=100, lr=0.001):
#     # Weighted Cross-Entropy Loss
#     criterion_celw = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight.item()]))
#     criterion_foc = FocalLoss(alpha=1, gamma=2)
#     criterion_cel = nn.CrossEntropyLoss()
#     criterion_pr = WeightedPRAUCLoss(positive_weight=1)
#     criterion_rec  = RecallLoss()
    
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # L2 regularization
#     rec_reg=.05
    
#     for epoch in range(1, num_epochs + 1):
#         model.train()
#         epoch_loss = 0

#         for X_batch, y_batch in train_loader:
#             optimizer.zero_grad()
#             outputs = model(X_batch)
#             loss_pr = criterion_pr(outputs, y_batch.long())
#             loss_rec = criterion_rec(outputs, y_batch.long())
#             loss_celw = criterion_celw(outputs, y_batch.long())
#             loss_cel = criterion_cel(outputs, y_batch.long())
#             loss_foc = criterion_foc(outputs, y_batch.long())
            
#             fin_loss = loss_pr 
#             fin_loss = loss_rec 
#             fin_loss = loss_cel 
#             fin_loss = loss_foc 
#             fin_loss = loss_celw 

#             # recall_loss_term = rec_reg * loss_rec
#             # fin_loss = loss_pr + recall_loss_term
#             # fin_loss = loss_cel + recall_loss_term  
#             # fin_loss = loss_foc + recall_loss_term  
#             # fin_loss = loss_celw + recall_loss_term 



#             fin_loss.backward()
#             optimizer.step()
#             epoch_loss += fin_loss.item()

#         # Evaluate every 5 epochs
#         if epoch % 5 == 0 or epoch == num_epochs:
#             test_accuracy, precision, recall, auroc = evaluate_metrics(model, test_loader)
#             print(
#                 f"Epoch {epoch}/{num_epochs} - Train Loss: {epoch_loss / len(train_loader):.4f}, "
#                 f"Test Accuracy: {test_accuracy:.4f}, Precision: {precision:.4f}, "
#                 f"Recall: {recall:.4f}, AUROC: {auroc:.4f}"
#             )


# def evaluate_metrics(model, loader):
#     model.eval()
#     y_true = []
#     y_pred = []
#     y_probs = []

#     with torch.no_grad():
#         for X_batch, y_batch in loader:
#             outputs = model(X_batch)
#             probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability of the positive class
#             predictions = torch.argmax(outputs, axis=1)

#             y_true.extend(y_batch.numpy())
#             y_pred.extend(predictions.numpy())
#             y_probs.extend(probabilities.numpy())

#     # Calculate metrics
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, zero_division=0)
#     recall = recall_score(y_true, y_pred, zero_division=0)
#     auroc = roc_auc_score(y_true, y_probs)

#     return accuracy, precision, recall, auroc






# import csv
# import pandas as pd