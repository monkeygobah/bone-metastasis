from plotting import *
from utils import prep_data, split_data, get_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def run_log_reg(X_resampled_bone,y_resampled_bone,X_test_bone,y_test_bone):
    # Train Logistic Regression for bone metastasis
    lr_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=500)
    lr_model.fit(X_resampled_bone, y_resampled_bone)

    # Predict probabilities for bone metastasis
    y_pred_lr = lr_model.predict(X_test_bone)
    y_proba_lr = lr_model.predict_proba(X_test_bone)[:, 1]

    precisions, recalls, f1s, thresholds = get_metrics(y_proba_lr)
    prf_thresh(thresholds,
               precisions, 
               recalls, 
               f1s, 
               title = "Logistic Regression: Precision, Recall, and F1-Score vs. Threshold", 
               name = 'log_reg_prf_thresh.png'
               )
    
    auroc_cm(y_test_bone, y_proba_lr, verbose=True, 
             auroc_title = "ROC Curve with Optimal Threshold (Youden's Index)", 
             auroc_name = 'lr_auroc.png',
             cm_title = 'Logistic Regression: Confusion Matrix at Optimal Threshold',
             cm_name = 'lr_cm.png',
             model = 'log reg'
             )

    feat_imp(None,
             X_resampled_bone, 
             n=20,
             title = "Top 10 Feature Importance (Logistic Regression)",
             name = 'lr_feat_imp.png',
             model =lr_model )



def run_xgb(X_resampled_bone,y_resampled_bone,X_test_bone,y_test_bone,y_train_bone):
    # Train XGBoost model for bone metastasis
    xgb_model_bone = XGBClassifier(random_state=42, scale_pos_weight=len(y_train_bone[y_train_bone == 0]) / len(y_train_bone[y_train_bone == 1]))
    xgb_model_bone.fit(X_resampled_bone, y_resampled_bone)
    y_pred_bone = xgb_model_bone.predict(X_test_bone)
    y_proba_bone = xgb_model_bone.predict_proba(X_test_bone)[:, 1]
    precisions, recalls, f1s, thresholds = get_metrics(y_proba_bone)


    prf_thresh(thresholds,
               precisions, 
               recalls, 
               f1s, 
               title = "XGBoost: Precision, Recall, and F1-Score vs. Threshold", 
               name = 'xgboost_prf_thresh.png'
               )
    
    auroc_cm(y_test_bone, y_proba_bone, verbose=True, 
             auroc_title = "ROC Curve with Optimal Threshold (Youden's Index)", 
             auroc_name = 'xg_auroc.png',
             cm_title = 'XGBoost: Confusion Matrix at Optimal Threshold',
             cm_name = 'xg_cm.png',
             model = 'xgboost'
             )

    # Extract feature importance and sort indices by descending order
    feature_importances = xgb_model_bone.feature_importances_
    feat_imp(feature_importances,
             X_resampled_bone, 
             n=20,
             title = "Top 10 Feature Importance (XGBoost)",
             name = 'xg_feat_imp.png')
    return y_proba_bone



def run_rf(X_resampled_bone,y_resampled_bone,X_test_bone,y_test_bone):
    # Train Random Forest for bone metastasis
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=200, max_depth=7)
    rf_model.fit(X_resampled_bone, y_resampled_bone)
    y_pred_rf = rf_model.predict(X_test_bone)
    y_proba_rf = rf_model.predict_proba(X_test_bone)[:, 1]

    precisions, recalls, f1s, thresholds = get_metrics(y_proba_rf)
    prf_thresh(thresholds,
               precisions, 
               recalls, 
               f1s, 
               title = "Random Forest: Precision, Recall, and F1-Score vs. Threshold", 
               name = 'randfor_prf_thresh.png'
               )
    

    auroc_cm(y_test_bone, y_proba_rf, verbose=True, 
             auroc_title = "ROC Curve with Optimal Threshold (Youden's Index)", 
             auroc_name = 'rf_auroc.png',
             cm_title = 'Random Forest: Confusion Matrix at Optimal Threshold',
             cm_name = 'rf_cm.png',
             model = 'random forest'
             )



    # Extract feature importance and sort indices by descending order
    feature_importances_rf = rf_model.feature_importances_
    feat_imp(feature_importances_rf,
             X_resampled_bone, 
             n=20,
             title = "Top 10 Feature Importance (Random Forest)",
             name = 'rf_feat_imp.png')
    return y_proba_rf
