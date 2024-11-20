import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

# Define the CNN model
class CNN1D(nn.Module):
    def __init__(self, input_size):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * input_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_dataloaders(X_train_bone,X_test_bone,y_train_bone,y_test_bone):
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bone)
    X_test_scaled = scaler.transform(X_test_bone)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_bone.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_bone.values, dtype=torch.float32)

    # Calculate positive class weight for imbalanced datasets
    positive_samples = y_train_tensor.sum().item()
    negative_samples = len(y_train_tensor) - positive_samples
    pos_weight = negative_samples / positive_samples
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,pos_weight_tensor

def engine(X_train_bone,X_test_bone,y_train_bone,y_test_bone):
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,pos_weight_tensor = get_dataloaders(X_train_bone,X_test_bone,y_train_bone,y_test_bone)
    # Initialize the model, loss function, and optimizer
    input_size = X_train_tensor.shape[1]
    cnn_model = CNN1D(input_size)

    # Split the data into training and validation sets
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train_tensor, y_train_tensor, test_size=0.2, random_state=42, stratify=y_train_tensor
    )

    train_loop(cnn_model,X_train_main,y_train_main,pos_weight_tensor,X_val,y_val)
    test(cnn_model, y_test_tensor, X_test_tensor)

def train_loop(cnn_model,X_train_main,y_train_main,pos_weight_tensor,X_val,y_val):
    # Training loop with early stopping
    num_epochs = 100
    batch_size = 64
    patience = 10
    best_val_loss = float('inf')
    early_stop_counter = 0
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in range(num_epochs):
        cnn_model.train()
        permutation = torch.randperm(X_train_main.size(0))
        epoch_loss = 0.0

        for i in range(0, X_train_main.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train_main[indices], y_train_main[indices]

            optimizer.zero_grad()
            outputs = cnn_model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= X_train_main.size(0)

        # Validation
        cnn_model.eval()
        with torch.no_grad():
            val_outputs = cnn_model(X_val).squeeze()
            val_loss = criterion(val_outputs, y_val).item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_state = cnn_model.state_dict()
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print('Early stopping!')
                break


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