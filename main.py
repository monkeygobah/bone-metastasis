#Import Libraries
from utils import prep_data, split_data, cumulative_feature_importance
from plotting import *
from ml_models import *
from dl_model import engine
import torch
import numpy as np
import random
'''
Our objective is to develop predictive models with a recall rate of at 
least 90% for metastasis detection, ensuring clinical relevance for use 
by healthcare providers. The optimal threshold was chosen to meet this requirement.
We will assess the best-performing model at this threshold, prioritizing comparisons 
of precision, F1 score, and AUC value to evaluate its overall effectiveness.

'''




def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    data = prep_data(save=False)
    X_train_bone, X_test_bone, y_train_bone, \
        y_test_bone,X_resampled_bone, y_resampled_bone = split_data(data, drop_real_world = False, drop_columns_experiment=False, drop_missing=False, bone=False)

    corr_matrix(data)
    y_proba_bone, xg_imps,y_pred_xgb = run_xgb(X_resampled_bone,y_resampled_bone,X_test_bone,y_test_bone,y_train_bone)
    y_proba_lr, lr_imps,y_pred_lr   = run_log_reg(X_resampled_bone,y_resampled_bone,X_test_bone,y_test_bone)
    # y_proba_lasso, lasso_imps = run_lasso_log_reg(X_resampled_bone, y_resampled_bone, X_test_bone, y_test_bone,verbose=False)
    y_proba_rf,rf_imps,y_pred_rf   = run_rf(X_resampled_bone,y_resampled_bone,X_test_bone,y_test_bone)
    
    y_pred_mlp, y_proba_mlp = engine(X_train_bone,X_test_bone,y_train_bone,y_test_bone)


    plot_all_curves(y_test_bone, y_proba_bone,y_proba_rf,y_proba_lr,y_proba_mlp)
    

    y_pred_ensemble = np.logical_or.reduce([y_pred_xgb, y_pred_lr, y_pred_rf, np.array(y_pred_mlp)]).astype(int)

    ensemble_accuracy = accuracy_score(y_test_bone, y_pred_ensemble)
    ensemble_precision = precision_score(y_test_bone, y_pred_ensemble)
    ensemble_recall = recall_score(y_test_bone, y_pred_ensemble)
    ensemble_f1 = f1_score(y_test_bone, y_pred_ensemble)
    ensemble_auroc = roc_auc_score(y_test_bone, np.maximum.reduce([y_proba_bone, y_proba_lr, y_proba_rf,y_pred_mlp]))

    print(f"Ensemble Metrics:")
    print(f"Accuracy: {ensemble_accuracy}")
    print(f"Precision: {ensemble_precision}")
    print(f"Recall: {ensemble_recall}")
    print(f"F1 Score: {ensemble_f1}")
    print(f"AUROC: {ensemble_auroc}")


    models = [
        {'model_name': 'LR', 'top_features': lr_imps},
        {'model_name': 'XGBoost', 'top_features': xg_imps},
        {'model_name': 'Random Forest', 'top_features': rf_imps}]
    
    _,results = cumulative_feature_importance(models)
    import pandas as pd
    results_df = pd.DataFrame([results])
    results_df.to_csv('combined_feat_imps_BONE.csv')
    print(results)

if __name__ == '__main__':
    set_seed(42)
    main()


