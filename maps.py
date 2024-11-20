RENAME_MAPPING_1 = {
    'seer_combined_mets_at_dxbone_2010': 'bone_met',
    'seer_combined_mets_at_dxlung_2010': 'lung_met',
    'age_recode_with_single_ages_and_90': 'age',
    'sex': 'sex',
    'raceethnicity': 'ethnicity',
    'origin_recode_nhia_hispanic_nonhisp': 'hispanic',
    'icdo3_histbehav': 'icdo3_histbehav',
    'year_of_diagnosis': 'diagnosis_year',
    'year_of_followup_recode': 'followup_year',
    'reason_no_cancerdirected_surgery': 'surgery',
    'chemotherapy_recode_yes_nounk': 'chemo',
    'time_from_diagnosis_to_treatment_in_days_recode': 'diag_days_to_treatment',
    'afp_pretreatment_interpretation_recode_2010': 'afp_pretreatment',
    'tumor_size_over_time_recode_1988': 'tumor_size',
    'marital_status_at_diagnosis': 'marital_status',
    'median_household_income_inflation_adj_to_2022': 'income',
    'ruralurban_continuum_code': 'rural_urban',
    'combined_t': 'combined_t',
    'combined_n': 'combined_n'
}


# Define custom mappings for T and N staging
T_MAPPING = {
    'T0': 0,
    'TX': 0.5,
    'T1a': 1,
    'T1': 1.5,
    'T1b': 2,
    'T2': 3,
    'T3NOS': 4,
    'T3': 4,
    'T3a': 4.5,
    'T3b': 5,
    'T4': 6
}

N_MAPPING = {
    'N0': 0,
    'NX': 0.5,
    'N1': 1
}


# Define the mapping for income
INCOME_MAPPING = {
    'Unknown/missing/no match/Not 1990-2022': 0,
    '< $40,000': 1,
    '$40,000 - $44,999': 2,
    '$45,000 - $49,999': 3,
    '$50,000 - $54,999': 4,
    '$55,000 - $59,999': 5,
    '$60,000 - $64,999': 6,
    '$65,000 - $69,999': 7,
    '$70,000 - $74,999': 8,
    '$75,000 - $79,999': 9,
    '$80,000 - $84,999': 10,
    '$85,000 - $89,999': 11,
    '$90,000 - $94,999': 12,
    '$95,000 - $99,999': 13,
    '$100,000 - $109,999': 14,
    '$110,000 - $119,999': 15,
    '$120,000+': 16
}


RENAME_MAPPING = {
    'bone_met': 'Bone Metastasis',
    'lung_met': 'Lung Metastasis',
    'sex_Male': 'Male',
    'age': 'Age',
    'income': 'Income',
    'marital_status_Married (including common law)': 'Married',
    'rural_urban_Nonmetropolitan counties not adjacent to a metropolitan area': 'Rural Area',
    'followup_duration': 'Follow-up Duration',
    'diag_days_to_treatment': 'Days to Treatment',
    'tumor_size': 'Tumor Size',
    'combined_t': 'T Staging',
    'combined_n': 'N Staging',
    'surgery_Surgery performed': 'Surgery Performed',
    'chemo_Yes': 'Chemotherapy Performed',
    'afp_pretreatment_cleaned_2': 'Elevated AFP',
}



# New y-axis labels for the top features
BIG_RENAME_MAPPING = {
    'surgery_Surgery performed': 'Surgery Performed',
    'surgery_Not recommended': 'Surgery Not Recommended',
    'afp_pretreatment_cleaned_1': 'Pretreatment AFP Borderline',
    'surgery_Not recommended, contraindicated due to other cond; autopsy only (1973-2002)': 'Surgery Contraindicated due to other condition',
    'afp_pretreatment_cleaned_0': 'Pretreatment AFP Normal',
    'afp_pretreatment_cleaned_2': 'Pretreatment AFP Elevated',
    'tumor_size': 'Tumor Size',
    'age': 'Age',
    'diag_days_to_treatment': 'Days to Treatment',
    'combined_t': 'T Staging',
    'income': 'Income',
    'combined_n': 'N Staging',
    'followup_duration': 'Follow-up Duration',
    'marital_status_Married (including common law)': 'Married',
    'chemo_Yes': 'Chemotherapy Performed',
    'ethnicity_Filipino': 'Filipino',
    'ethnicity_White': 'White',
    'sex_Male': 'Male',
    'rural_urban_Counties in metropolitan areas of 250,000 to 1 million pop': 'Urban (250k-1M Population)',
    'rural_urban_Counties in metropolitan areas of lt 250 thousand pop': 'Small Metropolitan Area (<250k Population)',
    'icdo3_histbehav_clear cell type': 'ICD-O-3: Clear Cell Type',
    'surgery_Recommended but not performed, patient refused': 'Surgery Recommended, Not Performed, Patient Refused',
    'surgery_Recommended but not performed, unknown reason': 'Surgery Recommended, Not Performed, Unknown Reason',
    'marital_status_Single (never married)': 'Single, Never Married',
    'ethnicity_Chinese': 'Chinese',
    'hispanic_Spanish-Hispanic-Latino': 'Hispanic',
    'diagnosis_year': 'Diagnosis Year',
    'ethnicity_Black': 'Black',
    'followup_year': 'Follow-up Year',
    'marital_status_Widowed': 'Widowed',
    'ethnicity_Vietnamese (1988+)': 'Vietnamese',
    
}