import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, classification_report, confusion_matrix, roc_auc_score,precision_score, recall_score
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


def engine(X_train_bone, X_test_bone, y_train_bone, y_test_bone):
    # Get DataLoaders and class weight
    # from imblearn.over_sampling import RandomOverSampler

    # ros = RandomOverSampler(sampling_strategy='minority')
    
    # X_resampled, y_resampled = ros.fit_resample(X_train_bone, y_train_bone)
    
    balanced_loader, normal_loader, test_loader, pos_weight_tensor = get_dataloaders(
        X_train_bone, X_test_bone, y_train_bone, y_test_bone, batch_size=32
    )

    # Define model
    input_size = X_train_bone.shape[1]
    model = MLP(input_size)

    # Train the model
    train_loop(model, normal_loader, test_loader, pos_weight_tensor, num_epochs=100, lr=0.001)



def train_loop(model, train_loader, test_loader, pos_weight, num_epochs=100, lr=0.001):
    # Weighted Cross-Entropy Loss
    criterion_celw = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight.item()]))
    criterion_foc = FocalLoss(alpha=1, gamma=2)
    criterion_cel = nn.CrossEntropyLoss()
    criterion_pr = WeightedPRAUCLoss(positive_weight=1)
    criterion_rec  = RecallLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # L2 regularization
    rec_reg=.05
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            # loss_pr = criterion_pr(outputs, y_batch.long())
            # loss_rec = criterion_rec(outputs, y_batch.long())
            loss_celw = criterion_celw(outputs, y_batch.long())
            # loss_cel = criterion_cel(outputs, y_batch.long())
            # loss_foc = criterion_foc(outputs, y_batch.long())
            
            # recall_loss_term = rec_reg * loss_rec
            
            # fin_loss = loss_pr + recall_loss_term
            # fin_loss = loss_cel + recall_loss_term  # This does not work with recall term
            # fin_loss = loss_foc + recall_loss_term  # This does not work with recall term
            # fin_loss = loss_celw + recall_loss_term # This does not work with recall term
            
            # fin_loss = loss_pr 
            # fin_loss = loss_cel 
            # fin_loss = loss_foc 
            fin_loss = loss_celw 


            fin_loss.backward()
            optimizer.step()
            epoch_loss += fin_loss.item()

        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs:
            test_accuracy, precision, recall, auroc = evaluate_metrics(model, test_loader)
            print(
                f"Epoch {epoch}/{num_epochs} - Train Loss: {epoch_loss / len(train_loader):.4f}, "
                f"Test Accuracy: {test_accuracy:.4f}, Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, AUROC: {auroc:.4f}"
            )


def evaluate_metrics(model, loader):
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

    return accuracy, precision, recall, auroc


def test(cnn_model, y_test_tensor, X_test_tensor):
    # Load the best model state
    # cnn_model.load_state_dict(best_model_state)

    # Evaluate the model on the test set
    cnn_model.eval()
    with torch.no_grad():
        test_outputs = cnn_model(X_test_tensor).squeeze()
        y_pred_proba_cnn = torch.sigmoid(test_outputs).numpy()
        y_test_np = y_test_tensor.numpy()

    # Find the optimal threshold using the ROC curve
    fpr_cnn, tpr_cnn, thresholds_cnn = roc_curve(y_test_np, y_pred_proba_cnn)
    optimal_idx_cnn = np.argmax(tpr_cnn - fpr_cnn)
    optimal_threshold_cnn = thresholds_cnn[optimal_idx_cnn]
    print(f'Optimal Threshold: {optimal_threshold_cnn:.4f}')

    # Apply the optimal threshold
    y_pred_cnn = (y_pred_proba_cnn >= optimal_threshold_cnn).astype(int)

    # Print evaluation metrics
    print("Bone Metastasis Prediction with CNN:")
    print(classification_report(y_test_np, y_pred_cnn))
    print("ROC-AUC Score:", roc_auc_score(y_test_np, y_pred_proba_cnn))
    print("Confusion Matrix:\n", confusion_matrix(y_test_np, y_pred_cnn))

    # ROC Curve Plot for CNN
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_cnn, tpr_cnn, label=f"CNN ROC Curve (AUC = {roc_auc_score(y_test_np, y_pred_proba_cnn):.2f})", color="darkorange")
    plt.scatter(fpr_cnn[optimal_idx_cnn], tpr_cnn[optimal_idx_cnn], color="red", label=f"Optimal Threshold = {optimal_threshold_cnn:.4f}")
    plt.plot([0, 1], [0, 1], color="black", linestyle="--")
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("ROC Curve with Optimal Threshold (CNN - Bone Metastasis)", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Confusion Matrix Plot for CNN
    cm_cnn = confusion_matrix(y_test_np, y_pred_cnn)
    cm_cnn_percentage = cm_cnn / cm_cnn.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(
        cm_cnn,
        annot=False,  # Disable default annotations for manual addition
        fmt="d",
        cmap="Blues",
        xticklabels=["No Metastasis", "Metastasis"],
        yticklabels=["No Metastasis", "Metastasis"],
        cbar_kws={"label": "Counts"},
    )

    # Annotate Confusion Matrix
    for i in range(cm_cnn.shape[0]):
        for j in range(cm_cnn.shape[1]):
            text_color = "white" if i == 0 else "black"
            # Add raw count
            ax.text(j + 0.5, i + 0.4, f"{cm_cnn[i, j]}", ha="center", va="center", color=text_color, fontsize=12, weight="bold")
            # Add percentage below count
            ax.text(j + 0.5, i + 0.6, f"({cm_cnn_percentage[i, j]:.1f}%)", ha="center", va="center", color=text_color, fontsize=10)

    plt.title(f"Confusion Matrix at Optimal Threshold ({optimal_threshold_cnn:.4f})", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.tight_layout()
    plt.show()