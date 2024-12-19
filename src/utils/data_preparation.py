import numpy as np
import pandas as pd

def clean_loan_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess loan application data based on established cleaning steps.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw input DataFrame containing loan application data
    
    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame ready for model training/prediction
    """
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # 1. Fix Categorical Columns
    if 'term' in data.columns:
        data['term'] = data['term'].astype('category')
    
    if 'home_ownership' in data.columns:
        data['home_ownership'] = data['home_ownership'].fillna('OTHER')
        data['home_ownership'] = data['home_ownership'].astype('category')
    
    if 'purpose' in data.columns:
        data['purpose'] = data['purpose'].fillna(data['purpose'].mode()[0])
        data['purpose'] = data['purpose'].astype('category')
    
    # 2. Fix Percentage Columns
    if 'int_rate' in data.columns:
        data['int_rate'] = data['int_rate'].str.rstrip('%').astype('float')
    
    if 'revol_util' in data.columns:
        data['revol_util'] = data['revol_util'].str.rstrip('%').astype('float')
    
    # 3. Fix Employment Length
    if 'emp_length' in data.columns:
        def clean_emp_length(value):
            if pd.isna(value):
                return np.nan
            value = str(value).lower().strip()
            if '10+' in value:
                return 10
            elif '< 1' in value:
                return 0
            return int(value.split()[0])
        
        data['emp_length'] = data['emp_length'].apply(clean_emp_length)
    
    # 4. Drop Unnecessary Columns
    columns_to_drop = ['id', 'member_id', 'application_approved_flag', 'desc']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    
    # 5. Handle Credit Features
    data['no_credit_card_history'] = (
        data['total_bc_limit'].isnull() & 
        data['tot_hi_cred_lim'].isnull() & 
        data['bc_util'].isnull() & 
        data['percent_bc_gt_75'].isnull()
    ).astype(int)
    
    # Impute credit limit features
    for col in ['total_bc_limit', 'tot_hi_cred_lim']:
        if col in data.columns:
            median_val = data[data[col].notnull()][col].median()
            data[col] = data[col].fillna(data.apply(
                lambda x: 0 if x['no_credit_card_history'] == 1 else median_val, 
                axis=1
            ))
    
    # Impute utilization features
    for col in ['bc_util', 'percent_bc_gt_75']:
        if col in data.columns:
            mean_val = data[
                (data[col].notnull()) & 
                (data['total_bc_limit'] > 0)
            ][col].mean()
            data[col] = data[col].fillna(data.apply(
                lambda x: 0 if x['no_credit_card_history'] == 1 else mean_val,
                axis=1
            ))
    
    # 6. Handle Derogatory Feature
    if 'mths_since_last_major_derog' in data.columns:
        data['no_derog_history'] = data['mths_since_last_major_derog'].isnull().astype(int)
        data['mths_since_last_major_derog'] = data['mths_since_last_major_derog'].fillna(-1)
    
    # 7. Handle Inquiry Feature
    if 'mths_since_recent_inq' in data.columns:
        zero_inq_median = data[
            (data['inq_last_6mths'] == 0) & 
            (data['mths_since_recent_inq'].notnull())
        ]['mths_since_recent_inq'].median()
        
        medians_by_inq = data[
            (data['inq_last_6mths'] > 0) & 
            (data['mths_since_recent_inq'].notnull())
        ].groupby('inq_last_6mths')['mths_since_recent_inq'].median()
        
        data['mths_since_recent_inq'] = data.apply(
            lambda x: zero_inq_median if (pd.isnull(x['mths_since_recent_inq']) and x['inq_last_6mths'] == 0)
            else medians_by_inq[x['inq_last_6mths']] if pd.isnull(x['mths_since_recent_inq'])
            else x['mths_since_recent_inq'],
            axis=1
        )
    
    # 8. Handle Total Current Balance
    if 'tot_cur_bal' in data.columns:
        data.loc[data['no_credit_card_history'] == 1, 'tot_cur_bal'] = 0
        
        medians_by_purpose = data[
            (data['no_credit_card_history'] == 0) & 
            (data['tot_cur_bal'].notnull())
        ].groupby('purpose')['tot_cur_bal'].median()
        
        for purpose in data['purpose'].unique():
            mask = (
                (data['tot_cur_bal'].isnull()) & 
                (data['purpose'] == purpose) & 
                (data['no_credit_card_history'] == 0)
            )
            data.loc[mask, 'tot_cur_bal'] = medians_by_purpose[purpose]
    
    # 9. Handle Employment Length Missing Values
    if 'emp_length' in data.columns:
        data['emp_length_missing'] = data['emp_length'].isnull().astype(int)
        medians_by_ownership = data.groupby('home_ownership')['emp_length'].median()
        
        for ownership in data['home_ownership'].unique():
            mask = (data['emp_length'].isnull()) & (data['home_ownership'] == ownership)
            data.loc[mask, 'emp_length'] = medians_by_ownership[ownership]
    
    # 10. Handle Revolving Utilization
    if 'revol_util' in data.columns:
        median_val = data['revol_util'].median()
        data['revol_util'] = data['revol_util'].fillna(median_val)
    
    return data

def prepare_data_for_inference(new_df, preprocessing_params):
    """Prepare new data using saved preprocessing parameters"""
    df = new_df.copy()

    # 1. Categorical Encodings
    df['home_ownership_encoded'] = df['home_ownership'].map(preprocessing_params['home_ownership_risk'])
    # Purpose encoding
    def encode_purpose(purpose):
        purpose_groups = preprocessing_params['purpose_groups']
        if purpose in purpose_groups['low_risk']:
            return 0
        elif purpose in purpose_groups['medium_risk']:
            return 1
        else:
            return 2
    df['purpose_encoded'] = df['purpose'].apply(encode_purpose)
    df['term_encoded'] = df['term'].apply(lambda x: 0 if '36' in x else 1)

    # Drop original categorical columns
    df = df.drop(['home_ownership', 'purpose', 'term'], axis=1)

    # 2. Log transformations
    df['log_annual_inc'] = np.log1p(df['annual_inc'])
    df['log_loan_amt'] = np.log1p(df['loan_amnt'])
    df['log_dti'] = np.log1p(df['dti'])

    # 3. Standard scaling
    for feature in preprocessing_params['scale_features']:
        df[f'{feature}_scaled'] = preprocessing_params['scaler_dict'][feature].transform(df[[feature]])

    # 4. Special handling for derogatory marks
    mask = df['mths_since_last_major_derog'] >= 0
    df['mths_since_derog_scaled'] = df['mths_since_last_major_derog'].copy()
    df.loc[mask, 'mths_since_derog_scaled'] = preprocessing_params['derog_scaler'].transform(
        df.loc[mask, ['mths_since_last_major_derog']])

    # 5. Feature Engineering
    df['income_to_loan_ratio'] = df['log_annual_inc'] / df['log_loan_amt']
    df['int_rate_dti'] = df['int_rate_scaled'] * df['log_dti']
    df['risk_score'] = (df['int_rate_scaled'] + 
                       df['log_dti'] + 
                       df['revol_util_scaled'] - 
                       df['income_to_loan_ratio'])

    # 6. Drop original features
    df = df.drop(preprocessing_params['columns_to_drop'], axis=1)

    # 7. Ensure correct column order
    df = df[preprocessing_params['feature_names']]

    return df