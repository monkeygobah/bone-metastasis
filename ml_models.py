from plotting import *
from utils import prep_data, split_data, get_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, classification_report, confusion_matrix, roc_auc_score,precision_score, recall_score, accuracy_score
from sklearn.metrics import f1_score


import matplotlib.pyplot as plt
import numpy as np

def plot_easy_sample_distribution(y_proba, y_true, easy_positive_threshold=0.8, easy_negative_threshold=0.2):
    # Separate easy positives and negatives
    easy_positives = y_proba[(y_proba >= easy_positive_threshold) & (y_true == 1)]
    easy_negatives = y_proba[(y_proba <= easy_negative_threshold) & (y_true == 0)]
    hard_samples = y_proba[(y_proba > easy_negative_threshold) & (y_proba < easy_positive_threshold)]

    # Plot histograms
    plt.figure(figsize=(10, 6))
    plt.hist(easy_positives, bins=20, alpha=0.6, color='green', label='Easy Positives')
    plt.hist(easy_negatives, bins=20, alpha=0.6, color='blue', label='Easy Negatives')
    plt.hist(hard_samples, bins=20, alpha=0.6, color='orange', label='Hard Samples')

    plt.axvline(x=easy_positive_threshold, color='red', linestyle='--', label=f'Positive Threshold = {easy_positive_threshold}')
    plt.axvline(x=easy_negative_threshold, color='purple', linestyle='--', label=f'Negative Threshold = {easy_negative_threshold}')

    plt.title("Distribution of Predicted Probabilities", fontsize=14)
    plt.xlabel("Predicted Probability", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example usage with your `run_log_reg` function


def run_log_reg(X_resampled_bone, y_resampled_bone, X_test_bone, y_test_bone,verbose=False):
    # Train Logistic Regression for bone metastasis
    lr_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=500)
    lr_model.fit(X_resampled_bone, y_resampled_bone)

    # Predict probabilities and labels for bone metastasis
    y_pred_lr = lr_model.predict(X_test_bone)
    y_proba_lr = lr_model.predict_proba(X_test_bone)[:, 1]

    # Compute metrics
    precision = precision_score(y_test_bone, y_pred_lr)
    recall = recall_score(y_test_bone, y_pred_lr)
    accuracy = accuracy_score(y_test_bone, y_pred_lr)
    auroc = roc_auc_score(y_test_bone, y_proba_lr)
    f1 = f1_score(y_test_bone, y_pred_lr)

    # Print metrics
    print(f"Logistic Regression Metrics:")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, AUROC: {auroc:.4f}, F1: {f1:.4f}")

    # Generate plots
    # precisions, recalls, f1s, thresholds = get_metrics(y_proba_lr, y_test_bone)
    # prf_thresh(thresholds, precisions, recalls, f1s,
    #            title="Logistic Regression: Precision, Recall, and F1-Score vs. Threshold",
    #            name='log_reg_prf_thresh.png')

    # auroc_cm(y_test_bone, y_proba_lr, verbose=verbose,
    #          auroc_title="ROC Curve with Optimal Threshold (Youden's Index)",
    #          auroc_name='lr_auroc.png',
    #          cm_title='Logistic Regression: Confusion Matrix at Optimal Threshold',
    #          cm_name='lr_cm.png',
    #          model='log reg')

    # plot_easy_sample_distribution(y_proba_lr, y_test_bone, easy_positive_threshold=0.8, easy_negative_threshold=0.2)

    imps, tops = feat_imp(None, X_resampled_bone, n=20,
             title="Top 10 Feature Importance (Logistic Regression)",
             name='lr_feat_imp.png',
             model=lr_model,
             log_reg=True)

    return y_proba_lr, imps,y_pred_lr



def run_lasso_log_reg(X_resampled_bone, y_resampled_bone, X_test_bone, y_test_bone,verbose=False):
    # Train Logistic Regression with LASSO (L1 regularization) for bone metastasis
    lasso_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=500, penalty='l1', solver='liblinear', C=1.0)
    lasso_model.fit(X_resampled_bone, y_resampled_bone)

    # Predict probabilities and labels for bone metastasis
    y_pred_lasso = lasso_model.predict(X_test_bone)
    y_proba_lasso = lasso_model.predict_proba(X_test_bone)[:, 1]

    # Compute metrics
    precision = precision_score(y_test_bone, y_pred_lasso)
    recall = recall_score(y_test_bone, y_pred_lasso)
    accuracy = accuracy_score(y_test_bone, y_pred_lasso)
    auroc = roc_auc_score(y_test_bone, y_proba_lasso)
    f1 = f1_score(y_test_bone, y_pred_lasso)

    # Print metrics
    print(f"LASSO Logistic Regression Metrics:")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, AUROC: {auroc:.4f}, F1: {f1:.4f}")

    # # Generate plots
    # precisions, recalls, f1s, thresholds = get_metrics(y_proba_lasso, y_test_bone)
    # prf_thresh(thresholds, precisions, recalls, f1s,
    #            title="LASSO Logistic Regression: Precision, Recall, and F1-Score vs. Threshold",
    #            name='lasso_log_reg_prf_thresh.png')

    # auroc_cm(y_test_bone, y_proba_lasso, verbose=verbose,
    #          auroc_title="ROC Curve with Optimal Threshold (Youden's Index)",
    #          auroc_name='lasso_lr_auroc.png',
    #          cm_title='LASSO Logistic Regression: Confusion Matrix at Optimal Threshold',
    #          cm_name='lasso_lr_cm.png',
    #          model='lasso log reg')

    imps, tops = feat_imp(None, X_resampled_bone, n=20,
             title="Top 10 Feature Importance (LASSO Logistic Regression)",
             name='lasso_lr_feat_imp.png',
             model=lasso_model,
             log_reg=True)

    return y_proba_lasso, imps



def run_xgb(X_resampled_bone, y_resampled_bone, X_test_bone, y_test_bone, y_train_bone,verbose=False):
    # Train XGBoost model for bone metastasis
    xgb_model_bone = XGBClassifier(random_state=42, scale_pos_weight=len(y_train_bone[y_train_bone == 0]) / len(y_train_bone[y_train_bone == 1]))
    xgb_model_bone.fit(X_resampled_bone, y_resampled_bone)

    # Predict probabilities and labels
    y_pred_bone = xgb_model_bone.predict(X_test_bone)
    y_proba_bone = xgb_model_bone.predict_proba(X_test_bone)[:, 1]

    # Compute metrics
    precision = precision_score(y_test_bone, y_pred_bone)
    recall = recall_score(y_test_bone, y_pred_bone)
    accuracy = accuracy_score(y_test_bone, y_pred_bone)
    auroc = roc_auc_score(y_test_bone, y_proba_bone)
    f1 = f1_score(y_test_bone, y_pred_bone)

    # Print metrics
    print(f"XGBoost Metrics:")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, AUROC: {auroc:.4f}, F1: {f1:.4f}")

    # # Generate plots
    # precisions, recalls, f1s, thresholds = get_metrics(y_proba_bone, y_test_bone)
    # prf_thresh(thresholds, precisions, recalls, f1s,
    #            title="XGBoost: Precision, Recall, and F1-Score vs. Threshold",
    #            name='xgboost_prf_thresh.png')

    # auroc_cm(y_test_bone, y_proba_bone, verbose=verbose,
    #          auroc_title="ROC Curve with Optimal Threshold (Youden's Index)",
    #          auroc_name='xg_auroc.png',
    #          cm_title='XGBoost: Confusion Matrix at Optimal Threshold',
    #          cm_name='xg_cm.png',
            #  model='xgboost')

    feature_importances = xgb_model_bone.feature_importances_
    imps, tops = feat_imp(feature_importances, X_resampled_bone, n=20,
             title="Top 10 Feature Importance (XGBoost)",
             name='xg_feat_imp.png')

    return y_proba_bone, imps,y_pred_bone


def run_rf(X_resampled_bone, y_resampled_bone, X_test_bone, y_test_bone,verbose=False):
    # Train Random Forest for bone metastasis
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=200, max_depth=7)
    rf_model.fit(X_resampled_bone, y_resampled_bone)

    # Predict probabilities and labels
    y_pred_rf = rf_model.predict(X_test_bone)
    y_proba_rf = rf_model.predict_proba(X_test_bone)[:, 1]

    # Compute metrics
    precision = precision_score(y_test_bone, y_pred_rf)
    recall = recall_score(y_test_bone, y_pred_rf)
    accuracy = accuracy_score(y_test_bone, y_pred_rf)
    auroc = roc_auc_score(y_test_bone, y_proba_rf)
    f1 = f1_score(y_test_bone, y_pred_rf)

    # Print metrics
    print(f"Random Forest Metrics:")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, AUROC: {auroc:.4f}, F1: {f1:.4f}")

    # # Generate plots
    # precisions, recalls, f1s, thresholds = get_metrics(y_proba_rf, y_test_bone)
    # prf_thresh(thresholds, precisions, recalls, f1s,
    #            title="Random Forest: Precision, Recall, and F1-Score vs. Threshold",
    #            name='randfor_prf_thresh.png')

    # auroc_cm(y_test_bone, y_proba_rf, verbose=verbose,
    #          auroc_title="ROC Curve with Optimal Threshold (Youden's Index)",
    #          auroc_name='rf_auroc.png',
    #          cm_title='Random Forest: Confusion Matrix at Optimal Threshold',
    #          cm_name='rf_cm.png',
    #          model='random forest')

    feature_importances_rf = rf_model.feature_importances_
    imps, tops = feat_imp(feature_importances_rf, X_resampled_bone, n=20,
             title="Top 10 Feature Importance (Random Forest)",
             name='rf_feat_imp.png')

    return y_proba_rf, imps,y_pred_rf
