from maps import N_MAPPING, INCOME_MAPPING, RENAME_MAPPING, T_MAPPING
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, classification_report, confusion_matrix, roc_auc_score
from maps import N_MAPPING, INCOME_MAPPING, RENAME_MAPPING, T_MAPPING,BIG_RENAME_MAPPING
import os 
output_dir = 'FIGS'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def corr_matrix(data):
    # Subset the columns you want to analyze
    columns_of_interest = list(RENAME_MAPPING.keys())
    data_subset = data[columns_of_interest]

    # Compute the correlation matrix
    correlation_matrix = data_subset.corr()
    correlation_matrix = correlation_matrix.rename(index=RENAME_MAPPING, columns=RENAME_MAPPING)
    mask = np.tril(np.ones_like(correlation_matrix, dtype=bool), k=-1)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        correlation_matrix,
        mask=mask,  
        annot=True,  
        fmt=".2f",  
        cmap="coolwarm",  
        cbar=True,        
        square=True,      
        linewidths=0.5,
        xticklabels=True,
        yticklabels=False  
    )
    plt.gca().xaxis.tick_top()
    plt.xticks(rotation=45, ha='left', fontsize=10) 
    for idx, label in enumerate(correlation_matrix.index):
        plt.text(
            idx - 0.3,  
            idx + 0.5,        
            label,      
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=10,
            color='black',
            transform=plt.gca().transData
        )
    plt.title("Triangular Correlation Matrix of Features (Upper Triangle Only)", fontsize=16, pad=20)
    # plt.show()
    plt.savefig(os.path.join(output_dir,'corr_matrix_feats.png'), dpi=300)



def prf_thresh(thresholds,precisions, recalls, f1s, title = None, name = None):

    # Plot the metrics
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label="Precision", color="blue")
    plt.plot(thresholds, recalls, label="Recall", color="orange")
    plt.plot(thresholds, f1s, label="F1 Score", color="green")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(os.path.join(output_dir, name),dpi=300)

def auroc_cm(y_test_bone, y_proba_bone,verbose=True, auroc_title = None, auroc_name = None, cm_title=None, cm_name = None, model=None):
    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test_bone, y_proba_bone)

    # Find the optimal threshold using Youden's Index
    optimal_idx = np.argmax(tpr - fpr)  # Maximizing Sensitivity - (1 - Specificity)
    optimal_threshold = thresholds[optimal_idx]

    # Apply the optimal threshold
    y_pred_custom = (y_proba_bone >= optimal_threshold).astype(int)

    if verbose:
        print_metrics(optimal_threshold, y_test_bone,y_pred_custom, y_proba_bone,model)


    get_auroc(optimal_threshold, y_test_bone, y_proba_bone,fpr,tpr,optimal_idx, auroc_title, auroc_name)
    get_cm(y_test_bone, y_pred_custom, optimal_threshold,cm_title, cm_name)


def print_metrics(optimal_threshold, y_test_bone,y_pred_custom, y_proba_bone,model):
    # Evaluate performance at this threshold
    print(f"{model} Evaluation with optimal threshold = {optimal_threshold:.4f}:")
    print("Classification Report:")
    print(classification_report(y_test_bone, y_pred_custom))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_bone, y_pred_custom))
    print("ROC-AUC Score:", roc_auc_score(y_test_bone, y_proba_bone))


def get_auroc(optimal_threshold, y_test_bone, y_proba_bone,fpr,tpr,optimal_idx, auroc_title, auroc_name):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test_bone, y_proba_bone):.2f})', color='darkorange')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Optimal Threshold = {optimal_threshold:.4f}')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(auroc_title)
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(os.path.join(output_dir, auroc_name), dpi=300)


def get_cm(y_test_bone, y_pred_custom, optimal_threshold, cm_title, cm_name):
    # Confusion Matrix at Optimal Threshold
    cm = confusion_matrix(y_test_bone, y_pred_custom)
    cm_percentage = cm / cm.sum(axis=1, keepdims=True) * 100  # Convert to percentages

    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(
        cm,
        annot=False,  # Disable default annotations to add both counts and percentages manually
        fmt="d",
        cmap="Blues",
        xticklabels=['No Metastasis', 'Metastasis'],
        yticklabels=['No Metastasis', 'Metastasis'],
        cbar_kws={'label': 'Counts'}
    )

    # Add annotations for counts and percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Determine text color based on cell position
            text_color = "white" if i == 0 else "black"
            # Add raw count
            ax.text(j + 0.5, i + 0.4, f"{cm[i, j]}", 
                    ha="center", va="center", color=text_color, fontsize=12, weight="bold")
            # Add percentage below count
            ax.text(j + 0.5, i + 0.5, f"({cm_percentage[i, j]:.1f}%)", 
                    ha="center", va="center", color=text_color, fontsize=10)

    plt.title(f"{cm_title} ({optimal_threshold:.4f})", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(output_dir, cm_name), dpi=300)


def feat_imp(feature_importances,X_resampled_bone, n=20, title = None, name = None, log_reg = False, model = None):
    top_n = n

    if log_reg:
        if model == None:
            raise ValueError('Need model when calculating feature importances using logistic regression')
        else:
            coeffs = np.abs(model.coef_[0])  # Get absolute values of coefficients
            sorted_indices = np.argsort(coeffs)[::-1]  # Sort from highest to lowest
            top_indices_lr = sorted_indices[:top_n]
            top_features_lr = [X_resampled_bone.columns[i] for i in top_indices_lr]
            top_importances = coeffs[top_indices_lr]
            top_features_renamed = [BIG_RENAME_MAPPING.get(feature, feature) for feature in top_features_lr]


    else:
        sorted_indices = np.argsort(feature_importances)[::-1]  # Sort from highest to lowest
        # Get the top 10 features
        top_indices = sorted_indices[:top_n]
        top_features = [X_resampled_bone.columns[i] for i in top_indices]
        top_importances = feature_importances[top_indices]


        top_features_renamed = [BIG_RENAME_MAPPING.get(feature, feature) for feature in top_features]

    # Plot the top 10 feature importance as a horizontal bar plot
    plt.figure(figsize=(12, 8))  # Increased width for a longer graph
    plt.barh(top_features_renamed, top_importances, color="blue")
    plt.xlabel("Feature Importance", fontsize=12)
    plt.ylabel("Top Features", fontsize=12)
    plt.title(title, fontsize=14)
    plt.gca().invert_yaxis()  # Invert y-axis to show most important features on top

    # Add grid and ticks for better readability
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Set x-axis ticks to whole numbers (0.1, 0.2, etc.)
    plt.xticks(np.arange(0, round(max(top_importances) + 0.1, 1), 0.1))  # Dynamic ticks rounded to nearest 0.1

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(output_dir, name), dpi=300)



def plot_all_curves(y_test_bone, y_proba_bone,y_proba_rf,y_proba_lr):
    # Calculate ROC metrics and AUC scores for each model
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test_bone, y_proba_bone)
    auc_xgb = roc_auc_score(y_test_bone, y_proba_bone)

    fpr_rf, tpr_rf, _ = roc_curve(y_test_bone, y_proba_rf)
    auc_rf = roc_auc_score(y_test_bone, y_proba_rf)

    fpr_lr, tpr_lr, _ = roc_curve(y_test_bone, y_proba_lr)
    auc_lr = roc_auc_score(y_test_bone, y_proba_lr)

    # Plot all ROC curves on the same plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {auc_xgb:.2f})", color="darkorange", lw=2)
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.2f})", color="blue", lw=2)
    plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.2f})", color="green", lw=2)

    # Add a diagonal line for reference
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", lw=2)

    # Customize plot appearance
    plt.title("ROC Curves for Machine Learning Models", fontsize=16, fontweight="bold")
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Show plot
    # plt.show()
    plt.savefig(os.path.join(output_dir, 'all_aurocs.png'),dpi=300)