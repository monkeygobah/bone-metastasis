import pandas as pd
from maps import N_MAPPING, INCOME_MAPPING, RENAME_MAPPING, T_MAPPING
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


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

def fix_names(data):
    # Simplify column names
    data.columns = (
        data.columns
        .str.lower() 
        .str.strip()
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace(r'\s+', '_', regex=True)
    )
    # Rename the columns
    data.rename(columns=RENAME_MAPPING, inplace=True)

    # Extract the part after "Hepatocellular carcinoma"
    data['icdo3_histbehav'] = data['icdo3_histbehav'].str.extract(r'Hepatocellular carcinoma, (.*)')

    file_path = 'SEER_Cleaned.xlsx'
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
    print("Duplicate columns:", duplicate_columns)

    # Drop duplicates if needed
    data = data.loc[:, ~data.columns.duplicated()]

    return data

def prep_data():
    # Load dataset
    data = pd.read_excel(r'C:\Users\Admin\Desktop\SEER\SEER_Final.xlsx')
    data = combine_cols(data)
    data = fix_names(data)
    data = do_mapping(data)
    data = do_ohe(data)

    # Define the file path for saving the Excel file
    file_path = r'hot_encoding.xlsx'
    # Save the DataFrame as an Excel file
    data.to_excel(file_path, index=False)

    print(f"Data saved successfully to {file_path}")


def split_data(data):
    # Separate features and targets for bone metastasis
    X = data.drop(columns=['bone_met', 'lung_met'])
    y_bone = data['bone_met']


    # Split data for bone metastasis
    X_train_bone, X_test_bone, y_train_bone, y_test_bone = train_test_split(X, y_bone, test_size=0.2, random_state=42)


    # Impute missing values for all columns
    imputer = SimpleImputer(strategy='most_frequent')  # Use 'most_frequent' for categorical data; 'mean' or 'median' for numerical data
    X_train_bone = pd.DataFrame(imputer.fit_transform(X_train_bone), columns=X_train_bone.columns)
    X_test_bone = pd.DataFrame(imputer.transform(X_test_bone), columns=X_test_bone.columns)


    # Drop any remaining non-numeric columns if necessary
    data = data.select_dtypes(include=[np.number])

    # Apply SMOTE for both bone
    sm_bone = SMOTE(random_state=42)
    X_resampled_bone, y_resampled_bone = sm_bone.fit_resample(X_train_bone, y_train_bone)

    return X_train_bone, X_test_bone, y_train_bone, y_test_bone,X_resampled_bone, y_resampled_bone



def get_metrics(y_proba_bone):
    # Compute precision, recall, and F1 for various thresholds
    thresholds = np.linspace(0, 1, 100)
    precisions, recalls, f1s = [], [], []

    for threshold in thresholds:
        y_pred = (y_proba_bone >= threshold).astype(int)  # Corrected variable name
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_bone, y_pred, average='binary', zero_division=0)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return precisions, recalls, f1s, thresholds