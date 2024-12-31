import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler


# Creating lookup tablze


def create_lookup_table():
    # Create a DataFrame with the specified columns
    LookupTable = pd.DataFrame({
        'column': [],
        'original': [],
        'imputed': []
    })

    return LookupTable

# Initial Data Cleaning

# Tidying up the column names


def fix_column_name(col):
    return col.strip().lower().replace(' ', '_').replace('(', '').replace(')', '')


def clean_column_names(df):
    df.columns = [fix_column_name(col) for col in df.columns]
    return df

# Choosing Index Column


def choose_index_column(df):
    data_indexed = df.set_index('customer_id')
    return data_indexed


# Handling Incosistent Data

def standardize_type_column(df, column_name):
    """
    Standardizes the values in the specified column of the DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame containing the column to be standardized.
    column_name (str): The name of the column to be standardized.

    Returns:
    pd.DataFrame: The DataFrame with the standardized column.
    """
    mask_joint = df[column_name].str.contains('joint', case=False, na=False)
    mask_individual = df[column_name].str.contains(
        'individual', case=False, na=False)
    mask_direct = df[column_name].str.contains('direct', case=False, na=False)

    df.loc[mask_joint, column_name] = 'JOINT'
    df.loc[mask_individual, column_name] = 'INDIVIDUAL'
    df.loc[mask_direct, column_name] = 'DIRECT_PAY'

    return df


def encode_emp_length(dataframe, column_name):
    """
    Encodes the 'emp_length' column in the given DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the column to encode.
    column_name (str): The name of the column to encode.

    Returns:
    pd.DataFrame: The DataFrame with the encoded column.
    """
    # Remove ' years' and ' year'
    dataframe[column_name +
              '_encoded'] = dataframe[column_name].str.replace(
        ' years', '').str.replace(' year', '')

    # Replace '< 1' with 0 and '10+' with 10
    dataframe[column_name +
              '_encoded'] = dataframe[column_name + '_encoded'].replace(
        {'< 1': 0, '10+': 10})

    # Convert the column to numeric
    dataframe[column_name +
              '_encoded'] = pd.to_numeric(dataframe[column_name + '_encoded'])

    return dataframe


def emp_length_encode_lookup(LookupTable):
    # updating lookup table
    # Create a dictionary to map employment lengths to numeric values
    emp_length_mapping = {
        '< 1 year': 0,
        '1 year': 1,
        '2 years': 2,
        '3 years': 3,
        '4 years': 4,
        '5 years': 5,
        '6 years': 6,
        '7 years': 7,
        '8 years': 8,
        '9 years': 9,
        '10+ years': 10
    }

    emp_length_mapping_df = pd.DataFrame(
        list(emp_length_mapping.items()), columns=['original', 'imputed'])

    emp_length_mapping_df['column'] = 'emp_length'

    LookupTable = pd.concat(
        [LookupTable, emp_length_mapping_df], ignore_index=True)

    return LookupTable

# Function to map a number to its corresponding grade


def map_number_to_grade(number):
    grade_ranges = {
        'A': range(1, 6),
        'B': range(6, 11),
        'C': range(11, 16),
        'D': range(16, 21),
        'E': range(21, 26),
        'F': range(26, 31),
        'G': range(31, 36)
    }

    for grade, grade_range in grade_ranges.items():
        if number in grade_range:
            return grade
    return None


def encode_grades(df):

    df['letter_grade'] = df['grade'].apply(
        map_number_to_grade)

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()
    df['grade_encoded'] = label_encoder.fit_transform(
        df['letter_grade'])

    return df


def grades_lookup(df, LookupTable):
    # Define the grade ranges
    grade_ranges = {
        'A': range(1, 6),
        'B': range(6, 11),
        'C': range(11, 16),
        'D': range(16, 21),
        'E': range(21, 26),
        'F': range(26, 31),
        'G': range(31, 36)
    }

    # Add grade ranges to the lookup table
    for grade, grade_range in grade_ranges.items():
        for number in grade_range:
            LookupTable.loc[len(LookupTable)] = ['grade', number, grade]

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(df['letter_grade'])

    # Add the grade encoding to the lookup table
    # Get the mapping dictionary
    label_mapping = {index: label for index,
                     label in enumerate(label_encoder.classes_)}

    # Convert the label mapping to a DataFrame
    label_mapping_df = pd.DataFrame(
        list(label_mapping.items()), columns=['imputed', 'original'])

    # Add the 'column' column with the value 'letter_grade'
    label_mapping_df['column'] = 'letter_grade'

    # Reorder the columns to match the desired order
    label_mapping_df = label_mapping_df[['column', 'original', 'imputed']]
    label_mapping_df

    # Concatenate the new DataFrame with the existing lookup table DataFrame
    LookupTable = pd.concat(
        [LookupTable, label_mapping_df], ignore_index=True)

    return LookupTable


def handle_inconsistent_data(df):
    data_standard_type = standardize_type_column(df, 'type')

    data_emp_encoded = encode_emp_length(
        data_standard_type, 'emp_length')

    data_grade_encoded = encode_grades(data_emp_encoded)

    data_grade_encoded['term'] = data_grade_encoded['term'].str.strip()

    return data_grade_encoded


def inconsistent_data_lookup(df, LookupTable):
    LookupTable = emp_length_encode_lookup(LookupTable)
    LookupTable = grades_lookup(df, LookupTable)

    return LookupTable


def create_datetime_issue_date(df):
    df['issue_date_datetime'] = pd.to_datetime(
        df['issue_date'], errors='coerce')
    return df


def data_integration_bonus(df):

    # Define the endpoint and headers
    url = "https://us-states.p.rapidapi.com/basic"
    headers = {
        'x-rapidapi-host': "us-states.p.rapidapi.com",
        'x-rapidapi-key': "0ddb3c5aefmshbfa1bc19cc83ae4p1d4045jsn33de395602a2"
    }

    # Make the GET request
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        state_data = response.json()

        # Convert the fintech_df to a DataFrame
        state_df = pd.DataFrame(state_data)
    else:
        print(response.status_code)

    # Keep only the 'name' and 'postal_code' columns
    state_df_filtered = state_df.filter(['name', 'postal'])
    state_df_filtered.columns = ['state_name', 'state']

    # Merge the data_indexed DataFrame with the state_df_filtered DataFrame on the 'addr_state' column
    data_with_state = df.merge(
        state_df_filtered, on='state', how='left').set_index(df.index)

    return data_with_state

# Cleaning Data


def impute_description(df):
    data_description_imputed = df.copy()
    data_description_imputed['description'] = data_description_imputed['description'].fillna(
        data_description_imputed['purpose'])

    return data_description_imputed


def impute_emp_title(df):
    imputationString = 'Missing'
    df['emp_title'] = df['emp_title'].fillna(
        imputationString)
    return df


def emp_title_lookup(LookupTable):
    imputationString = 'Missing'

    # Add the 'emp_title' column with original NaN and imputed 'Missing' to the LookupTable
    lookup_update = pd.DataFrame({
        'column': ['emp_title'],
        'original': [np.nan],
        'imputed': [imputationString]
    })

    LookupTable = pd.concat([LookupTable, lookup_update], ignore_index=True)
    return LookupTable


def impute_int_rate(df):
    # Impute missing int_rate values with the median of the corresponding grade
    df['int_rate'] = df.groupby(
        'grade')['int_rate'].transform(lambda x: x.fillna(x.median()))

    imputation_int_rate = df.groupby(
        'letter_grade')['int_rate'].median().reset_index()

    imputation_int_rate.columns = ['letter_grade', 'median_int_rate']
    imputation_int_rate.to_csv('data/imputation_int_rate.csv', index=False)

    return df


def impute_emp_length(df):
    imputationCol = 'emp_length'
    OriginalVal = np.nan
    imputationVal = '< 1 year'

    df[imputationCol] = df[imputationCol].fillna(
        imputationVal)

    imputationCol = 'emp_length_encoded'
    OriginalVal = np.nan
    imputationVal = 0

    df[imputationCol] = df[imputationCol].fillna(
        imputationVal)

    return df


def impute_annual_inc_joint(df):
    df['annual_inc_joint'] = df['annual_inc_joint'].fillna(
        df['annual_inc'])
    return df


def emp_length_lookup(LookupTable):

    imputationCol = 'emp_length'
    OriginalVal = np.nan
    imputationVal = '< 1 year'
    # Add the 'emp_title' column with original NaN and imputed 'Missing' to the LookupTable
    lookup_update = pd.DataFrame({
        'column': [imputationCol],
        'original': [OriginalVal],
        'imputed': [imputationVal]
    })

    LookupTable = pd.concat([LookupTable, lookup_update], ignore_index=True)

    # emp length encoded lookup
    imputationCol = 'emp_length_encoded'
    OriginalVal = np.nan
    imputationVal = 0
    # Add the 'emp_title' column with original NaN and imputed 'Missing' to the LookupTable
    lookup_update = pd.DataFrame({
        'column': [imputationCol],
        'original': [OriginalVal],
        'imputed': [imputationVal]
    })

    LookupTable = pd.concat([LookupTable, lookup_update], ignore_index=True)
    LookupTable

    return LookupTable


def handle_missing_values(df):
    df = impute_description(df)
    df = impute_emp_title(df)
    df = impute_int_rate(df)
    df = impute_emp_length(df)
    df = impute_annual_inc_joint(df)

    return df


def missing_values_lookup(LookupTable):
    LookupTable = emp_title_lookup(LookupTable)
    LookupTable = emp_length_lookup(LookupTable)
    return LookupTable


def calculate_monthly_installments(row):
    principal = row['loan_amount']
    annual_rate = row['int_rate']
    term_months = int(row['term'].split()[0])

    monthly_rate = annual_rate / 12
    numerator = principal * monthly_rate * ((1 + monthly_rate) ** term_months)
    denominator = ((1 + monthly_rate) ** term_months) - 1
    return numerator / denominator if denominator != 0 else 0


def add_installment_per_month(df):
    df['installment_per_month'] = df.apply(
        calculate_monthly_installments, axis=1)
    return df


def calculate_iqr_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_mask = (column < lower_bound) | (column > upper_bound)
    outliers = column[outliers_mask]
    outlier_percentage = len(outliers) / len(column) * 100
    return outlier_percentage


def log_transform(df, numeric_cols):
    # Apply log1p transform to all numeric columns
    data_log_transformed = df.copy()

    for col in numeric_cols:
        data_log_transformed[col] = np.log1p(data_log_transformed[col])

    return data_log_transformed


def handle_outliers(df, numeric_cols):
    df = log_transform(df, numeric_cols)
    return df

# Data Transformation and Feature Engineering

# Adding Columns


def add_month_number(df):
    df['month_number'] = df['issue_date_datetime'].dt.month
    return df


def add_salary_can_cover(df):
    df['salary_can_cover'] = (
        df['annual_inc_joint'] >= df['loan_amount'])
    return df


def add_remaining_columns(df):
    df = add_month_number(df)
    df = add_salary_can_cover(df)
    return df


# Encoding

def onehot_encoding(df):
    columns_to_encode = ['home_ownership',
                         'verification_status', 'type', 'purpose']

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder()

    # Perform one-hot encoding on the specified columns
    encoded_columns = encoder.fit_transform(
        df[columns_to_encode]).toarray()

    # Get the feature names for the encoded columns
    encoded_feature_names = encoder.get_feature_names_out(
        columns_to_encode)

    # Create a DataFrame with the encoded columns
    encoded_df = pd.DataFrame(
        encoded_columns, columns=encoded_feature_names).set_index(df.index)

    # Concatenate the original DataFrame with the encoded DataFrame
    df = pd.concat([df.drop(
        columns=columns_to_encode), encoded_df], axis=1)

    return df


def encode_term(df):
    df['term'] = df['term'].str.replace(' months', '').astype(int)
    return df


def encode_term_lookup(LookupTable):
    # Create a dictionary to map employment lengths to numeric values
    term_mapping = {
        '36 months': 36,
        '60 months': 60,
    }

    term_mapping_df = pd.DataFrame(
        list(term_mapping.items()), columns=['original', 'imputed'])

    term_mapping_df['column'] = 'term'

    LookupTable = pd.concat(
        [LookupTable, term_mapping_df], ignore_index=True)
    return LookupTable


def encode_loan_status(df):

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Fit and transform the loan_status column
    df['loan_status'] = label_encoder.fit_transform(
        df['loan_status'])

    label_mapping = dict(
        zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    return df


def encode_loan_status_lookup(df, LookupTable):
    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Fit and transform the loan_status column
    label_encoder.fit_transform(
        df['loan_status'])

    label_mapping = dict(
        zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    # Get the mapping dictionary
    label_mapping = {index: label for index,
                     label in enumerate(label_encoder.classes_)}

    # Convert the label mapping to a DataFrame
    label_mapping_df = pd.DataFrame(
        list(label_mapping.items()), columns=['imputed', 'original'])

    # Add the 'column' column with the value 'loan_status'
    label_mapping_df['column'] = 'loan_status'

    # Reorder the columns to match the desired order
    label_mapping_df = label_mapping_df[['column', 'imputed', 'original']]

    # Concatenate the new DataFrame with the existing lookup table DataFrame
    LookupTable = pd.concat(
        [LookupTable, label_mapping_df], ignore_index=True)
    return LookupTable


def label_encoding(df):
    df = encode_term(df)
    df = encode_loan_status(df)

    # remove unencoded columns
    df = df.drop(columns=['emp_length', 'issue_date'])
    df.rename(columns={'emp_length_encoded': 'emp_length',
                       'issue_date_datetime': 'issue_date'}, inplace=True)
    return df


def label_encoding_lookup(df, LookupTable):
    LookupTable = encode_term_lookup(LookupTable)
    LookupTable = encode_loan_status_lookup(df, LookupTable)

    return LookupTable


# Normalization
def normalize_data(df, numeric_cols):
    scaler = MinMaxScaler()

    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(
        df_scaled[numeric_cols])

    # Get min/max values before scaling
    min_vals = df[numeric_cols].min()
    max_vals = df[numeric_cols].max()

    # Create DataFrame with min/max values
    scaling_params = pd.DataFrame({
        'column': numeric_cols,
        'min': min_vals,
        'max': max_vals
    })

    # Save scaling parameters
    scaling_params.to_csv('data/scaling_params.csv', index=False)

    return df_scaled


def clean(fintech_df):
    fintech_df = clean_column_names(fintech_df)
    fintech_df = choose_index_column(fintech_df)
    LookupTable = create_lookup_table()

    fintech_df = handle_inconsistent_data(fintech_df)
    LookupTable = inconsistent_data_lookup(fintech_df, LookupTable)

    fintech_df = create_datetime_issue_date(fintech_df)

    # fintech_df = data_integration_bonus(fintech_df)

    fintech_df = handle_missing_values(fintech_df)
    LookupTable = missing_values_lookup(LookupTable)

    fintech_df = add_installment_per_month(fintech_df)

    numeric_cols = fintech_df.select_dtypes(include=['number']).drop(
        columns=['grade', 'grade_encoded', 'loan_id', 'emp_length_encoded']).columns

    fintech_df = handle_outliers(fintech_df, numeric_cols)

    fintech_df = add_remaining_columns(fintech_df)

    fintech_df = onehot_encoding(fintech_df)

    LookupTable = label_encoding_lookup(fintech_df, LookupTable)
    fintech_df = label_encoding(fintech_df)

    fintech_df = normalize_data(fintech_df, numeric_cols)

    return fintech_df, LookupTable


def clean_row(json):

    row = pd.DataFrame([json])  # Convert the JSON row to a DataFrame

    row = clean_column_names(row)  # Clean the column names
    row = choose_index_column(row)  # Choose the index column

    row = standardize_type_column(row, 'type')  # Standardize the 'type' column
    # Create the 'issue_date_datetime' column
    row = create_datetime_issue_date(row)

    # Encode the 'emp_length' column
    row = encode_emp_length(row, 'emp_length')

    # Encode the 'grade' column
    row['letter_grade'] = row['grade'].apply(
        map_number_to_grade)

    LookupTable = pd.read_csv(
        'data/lookup_fintech_data_MET_P1_52_16824.csv')

    encoded_value = LookupTable.loc[
        (LookupTable['column'] == 'letter_grade') &
        (LookupTable['original'] == row['letter_grade'].iloc[0])
    ].values[0][2]

    # Map the letter grade to encoded value
    row['grade_encoded'] = encoded_value
    row['grade_encoded'] = row['grade_encoded'].astype(
        int)  # Convert the 'grade_encoded' column to int

    # Encode Loan Status
    encoded_value = LookupTable.loc[
        (LookupTable['column'] == 'loan_status') &
        (LookupTable['original'] == row['loan_status'].iloc[0])
    ].values[0][2]

    row['loan_status'] = encoded_value
    row['loan_status'] = row['loan_status'].astype(int)

    row = impute_description(row)  # Impute the 'description' column
    row = impute_emp_title(row)  # Impute the 'emp_title' column
    row = impute_emp_length(row)  # Impute the 'emp_length' column
    row = impute_annual_inc_joint(row)  # Impute the 'annual_inc_joint' column

    # impute int_rate
    int_rate_imputation = pd.read_csv('data/imputation_int_rate.csv')
    int_rate = int_rate_imputation.loc[(
        int_rate_imputation['letter_grade'] == row['letter_grade'].iloc[0])]['median_int_rate'].values[0]

    row['int_rate'] = row['int_rate'].fillna(int_rate)

    row = add_month_number(row)  # Add the month_number column

    # TODO: uncomment this before submission
    # row = data_integration_bonus(row)  # Perform data integration

    # Add the 'installment_per_month' column
    row = add_installment_per_month(row)
    row = add_salary_can_cover(row)  # Add the 'salary_can_cover' column

    row = encode_term(row)  # Encode the 'term' column

    numeric_cols = row.select_dtypes(include=['number']).drop(
        columns=['grade', 'grade_encoded', 'loan_id', 'emp_length_encoded', 'month_number', 'term', 'loan_status']).columns

    row = handle_outliers(row, numeric_cols)  # Handle outliers
    row = onehot_encoding(row)  # Perform one-hot encoding

    # scale the numeric columns
    scaling_params = pd.read_csv('data/scaling_params.csv')
    for col in numeric_cols:
        min_val = scaling_params.loc[scaling_params['column']
                                     == col]['min'].values[0]
        max_val = scaling_params.loc[scaling_params['column']
                                     == col]['max'].values[0]
        row[col] = (row[col] - min_val) / (max_val - min_val)

    # remove unencoded columns
    row = row.drop(columns=['emp_length', 'issue_date'])
    row.rename(columns={'emp_length_encoded': 'emp_length',
                        'issue_date_datetime': 'issue_date'}, inplace=True)

    main_df = pd.read_parquet(
        'data/fintech_data_MET_P1_52_16824_clean.parquet')

    required_columns = main_df.columns

    # Add missing columns with 0.0
    for col in required_columns:
        if col not in row.columns:
            row[col] = 0.0

    return row
