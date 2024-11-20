#Import Libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from maps import N_MAPPING, INCOME_MAPPING, RENAME_MAPPING, T_MAPPING,BIG_RENAME_MAPPING
from utils import prep_data, split_data, get_metrics
from plotting import *
from ml_models import *
from dl_model import engine
'''
Our objective is to develop predictive models with a recall rate of at 
least 90% for metastasis detection, ensuring clinical relevance for use 
by healthcare providers. The optimal threshold was chosen to meet this requirement.
We will assess the best-performing model at this threshold, prioritizing comparisons 
of precision, F1 score, and AUC value to evaluate its overall effectiveness.

'''



def main():
    data = prep_data(save=False)
    corr_matrix(data)
    X_train_bone, X_test_bone, y_train_bone, \
        y_test_bone,X_resampled_bone, y_resampled_bone = split_data(data)
    
    y_proba_bone = run_xgb(X_resampled_bone,y_resampled_bone,X_test_bone,y_test_bone,y_train_bone)
    y_proba_lr   = run_log_reg(X_resampled_bone,y_resampled_bone,X_test_bone,y_test_bone)
    y_proba_rf   = run_rf(X_resampled_bone,y_resampled_bone,X_test_bone,y_test_bone)

    plot_all_curves(y_test_bone, y_proba_bone,y_proba_rf,y_proba_lr)
    engine(X_train_bone,X_test_bone,y_train_bone,y_test_bone)



if __name__ == '__main__':
    main()


