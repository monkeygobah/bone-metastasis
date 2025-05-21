import pandas as pd
from maps import N_MAPPING, INCOME_MAPPING, RENAME_MAPPING, T_MAPPING,RENAME_MAPPING_1
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os
import matplotlib.pyplot as plt


def combine_cols(data):
    # Combine Derived AJCC T columns
    data['Combined_T'] = data['Derived AJCC T, 7th ed (2010-2015)'].where(
        data['Derived AJCC T, 7th ed (2010-2015)'] != 'Blank(s)', 
        data['Derived EOD 2018 T (2018+)']
    )

    # Combine Derived AJCC N columns
    data['Combined_N'] = data['Derived AJCC N, 7th ed (2010-2015)'].where(
        data['Derived AJCC N, 7th ed (2010-2015)'] != 'Blank(s)', 
        data['Derived EOD 2018 N (2018+)']
    )

    # Drop the original columns
    data.drop(columns=[
        'Derived AJCC T, 7th ed (2010-2015)', 
        'Derived EOD 2018 T (2018+)', 
        'Derived AJCC N, 7th ed (2010-2015)', 
        'Derived EOD 2018 N (2018+)'
    ], inplace=True)

    return data

def fix_names(data, save=False):
    # Simplify column names
    data.columns = (
        data.columns
        .str.lower() 
        .str.strip()
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace(r'\s+', '_', regex=True)
    )
    # Rename the columns
    data.rename(columns=RENAME_MAPPING_1, inplace=True)

    # Extract the part after "Hepatocellular carcinoma"
    data['icdo3_histbehav'] = data['icdo3_histbehav'].str.extract(r'Hepatocellular carcinoma, (.*)')

    file_path = 'SEER_Cleaned.xlsx'
    if save:
        data.to_excel(file_path, index=False)

    return data

def do_mapping(data):
    # Apply the mappings
    data['combined_t'] = data['combined_t'].map(T_MAPPING)
    data['combined_n'] = data['combined_n'].map(N_MAPPING)

    # Process 'age' column
    data['age'] = data['age'].str.extract('(\d+)').astype(int)
    # Calculate follow-up duration
    data['followup_duration'] = data['followup_year'] - data['diagnosis_year']

    # Replace 'Unable to calculate' with NaN
    data['diag_days_to_treatment'] = pd.to_numeric(data['diag_days_to_treatment'], errors='coerce')

    # Replace non-numeric values in tumor_size
    data['tumor_size'] = pd.to_numeric(data['tumor_size'], errors='coerce')

    # Apply the mapping to the income column
    data['income'] = data['income'].map(INCOME_MAPPING)
    return data



def do_ohe(data):
    # Clean and one-hot encode other categorical columns
    data['afp_pretreatment_cleaned'] = data['afp_pretreatment'].replace({
        'Positive/elevated': '2',
        'Negative/normal; within normal limits': '0',
        'Borderline; undetermined if positive or negative': '1',
        'Not documented; Not assessed or unknown if assessed': '-1',
        'Test ordered, results not in chart': '-1',
        'Not applicable: Information not collected for this case': '-1'
    })

    # Drop the original afp_pretreatment column
    data = data.drop(columns=['afp_pretreatment'])

    # Map binary columns to 0 and 1
    binary_cols = ['bone_met', 'lung_met']
    for col in binary_cols:
        data[col] = data[col].map({'Yes': 1, 'No': 0})


    # Include binary_cols in one-hot encoding
    categorical_cols = ['sex', 'ethnicity', 'hispanic', 'icdo3_histbehav', 'surgery', 'sex',
                        'chemo', 'afp_pretreatment_cleaned', 'marital_status', 'rural_urban'
                        ]

    # One-hot encode all categorical columns
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Check for duplicate column names
    duplicate_columns = data.columns[data.columns.duplicated()].tolist()

    # Drop duplicates if needed
    data = data.loc[:, ~data.columns.duplicated()]

    return data

def prep_data(save=False):
    # Load dataset
    data = pd.read_excel(r'SEER_Final.xlsx')
    data = combine_cols(data)
    data = fix_names(data, save=save)
    data = do_mapping(data)
    data = do_ohe(data)
    
    if save:
        file_path = r'hot_encoding.xlsx'
        data.to_excel(file_path, index=False)
        print(f"Data saved successfully to {file_path}")



    # # Example usage
    # features = [
    #     'Surgery Not Recommended', 'ethnicity_Thai (1994+)', 'ethnicity_Asian Indian or Pakistani, NOS (1988+)', 
    #     'Surgery Recommended, Not Performed, Unknown Reason', 'ethnicity_Pacific Islander, NOS (1991+)', 
    #     'Pretreatment AFP Borderline', 'Surgery Recommended, Not Performed, Patient Refused', 
    #     'Surgery Contraindicated due to other condition', 'icdo3_histbehav_scirrhous', 'N Staging', 
    #     'ethnicity_Pakistani (2010+)', 'ethnicity_Samoan (1991+)', 'icdo3_histbehav_spindle cell variant', 
    #     'surgery_Recommended, unknown if performed', 'Pretreatment AFP Normal', 'ethnicity_Laotian (1988+)', 
    #     'ethnicity_Micronesian, NOS (1991+)', 'ethnicity_Other Asian (1991+)', 'icdo3_histbehav_fibrolamellar', 
    #     'Follow-up Duration'
    # ]
    # plot_feature_distributions(features, data, label_column="bone_met")
    return data





def split_data(data, drop_real_world=False, drop_columns_experiment=False, drop_missing=False ,bone=True):
    # Separate features and targets for bone metastasis
    X = data.drop(columns=['bone_met', 'lung_met'])

    # Drop real-world columns if specified
    if drop_real_world:
        X = X.drop(columns=[
            'surgery_Not recommended',
            "surgery_Not recommended, contraindicated due to other cond; autopsy only (1973-2002)",
            "surgery_Recommended but not performed, patient refused",
            "surgery_Recommended but not performed, unknown reason",
            "surgery_Recommended, unknown if performed",
            'surgery_Surgery performed',
            'surgery_Unknown; death certificate; or autopsy only (2003+)'
        ])

    if drop_columns_experiment:
        X = X.drop(columns=[
            "rural_urban_Counties in metropolitan areas of 250,000 to 1 million pop",
            "rural_urban_Counties in metropolitan areas of lt 250 thousand pop",
            "rural_urban_Nonmetropolitan counties adjacent to a metropolitan area",
            "rural_urban_Nonmetropolitan counties not adjacent to a metropolitan area",
            "rural_urban_Unknown/missing/no match/Not 1990-2022",
            "ethnicity_Asian Indian (2010+)",
            "ethnicity_Asian Indian or Pakistani, NOS (1988+)",
            "ethnicity_Black",
            "ethnicity_Chamorran (1991+)",
            "ethnicity_Chinese",
            "ethnicity_Fiji Islander (1991+)",
            "ethnicity_Filipino",
            "ethnicity_Guamanian, NOS (1991+)",
            "ethnicity_Hawaiian",
            "ethnicity_Hmong (1988+)",
            "ethnicity_Japanese",
            "ethnicity_Kampuchean (1988+)",
            "ethnicity_Korean (1988+)",
            "ethnicity_Laotian (1988+)",
            "ethnicity_Micronesian, NOS (1991+)",
            "ethnicity_Other",
            "ethnicity_Other Asian (1991+)",
            "ethnicity_Pacific Islander, NOS (1991+)",
            "ethnicity_Pakistani (2010+)",
            "ethnicity_Samoan (1991+)",
            "ethnicity_Thai (1994+)",
            "ethnicity_Tongan (1991+)",
            "ethnicity_Unknown",
            "ethnicity_Vietnamese (1988+)",
            "ethnicity_White"
        ])

    if drop_missing:
        X = X.drop(columns=[
            "diag_days_to_treatment",
            "tumor_size",

        ])



    # Select target column based on the `bone` parameter
    if bone:
        y_bone = data['bone_met']
    else:
        y_bone = data['lung_met']


    # Split data for bone metastasis
    X_train_bone, X_test_bone, y_train_bone, y_test_bone = train_test_split(X, y_bone, test_size=0.2, random_state=42)


    # # Impute missing values for all columns
    # imputer = SimpleImputer(strategy='most_frequent')  
    # X_train_bone = pd.DataFrame(imputer.fit_transform(X_train_bone), columns=X_train_bone.columns)
    # X_test_bone = pd.DataFrame(imputer.transform(X_test_bone), columns=X_test_bone.columns)

    X_train_bone, X_test_bone = impute_data(X_train_bone, X_test_bone, strategy='iterative')  

    # Drop any remaining non-numeric columns if necessary
    data = data.select_dtypes(include=[np.number])

    sm_bone = SMOTE(random_state=42)
    X_resampled_bone, y_resampled_bone = sm_bone.fit_resample(X_train_bone, y_train_bone)

    return X_train_bone, X_test_bone, y_train_bone, y_test_bone,X_resampled_bone, y_resampled_bone



def impute_data(X_train, X_test, strategy='knn'):
    # Separate numerical and categorical columns

    numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X_train.select_dtypes(exclude=['float64', 'int64']).columns  

    if len(numerical_cols) > 0:
        if strategy == 'knn':
            imputer_num = KNNImputer(n_neighbors=5)
        elif strategy == 'iterative':
            imputer_num = IterativeImputer(max_iter=10, random_state=42)
        else: 
            imputer_num = SimpleImputer(strategy='mean')

        X_train[numerical_cols] = imputer_num.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = imputer_num.transform(X_test[numerical_cols])

    if len(categorical_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X_train[categorical_cols] = imputer_cat.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = imputer_cat.transform(X_test[categorical_cols])

    return X_train, X_test



def get_metrics(y_proba_bone,y_test_bone):
    thresholds = np.linspace(0, 1, 100)
    precisions, recalls, f1s = [], [], []

    for threshold in thresholds:
        y_pred = (y_proba_bone >= threshold).astype(int)  # Corrected variable name
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_bone, y_pred, average='binary', zero_division=0)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return precisions, recalls, f1s, thresholds



def cumulative_feature_importance(models, thresholds=[3, 5, 10, 15, 20]):
    results = {}
    conserved_features = {}

    for threshold in thresholds:
        if not isinstance(threshold, int):
            raise ValueError(f"Threshold value must be an integer, got {threshold}")
        
        # Ensure top_features is a list and perform slicing safely
        top_features_sets = [set(model['top_features'][:int(threshold)]) for model in models]
        
        # Compute intersection of top features across all models
        common_features = set.intersection(*top_features_sets)
        
        # Store the result
        results[threshold] = len(common_features)
        conserved_features[threshold] = common_features  # Save the feature names for this threshold
    
    # Print the conserved features for each threshold
    # for threshold, features in conserved_features.items():
    #     print(f"\nFor threshold {threshold}, conserved features ({len(features)}):")
    #     for feature in features:
    #         print(f"- {feature}")

    return results, conserved_features





def plot_feature_distributions(features, data, label_column, output_dir="feature_distributions"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for feature in features:
        if feature not in data.columns:
            print(f"Feature '{feature}' not found in data.")
            continue
        
        # Group by the feature and label column, count occurrences
        grouped = data.groupby([feature, label_column]).size().unstack(fill_value=0)
        
        # Create a bar plot
        grouped.plot(kind='bar', figsize=(10, 6))
        plt.title(f"Distribution of {label_column} for {feature}", fontsize=14)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title=label_column, labels=['0', '1'])
        
        # Save the plot
        save_path = os.path.join(output_dir, f"{feature}_distribution.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    print(f"Plots saved in '{output_dir}'.")

