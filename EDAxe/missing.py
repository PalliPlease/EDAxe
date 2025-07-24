def remove_rows(data, columns=None, threshold=5):
    """
    Drop rows where specified (or all) columns have missing values,
    but only if the column's missing percentage is below the threshold.

    Parameters:
        data (pd.DataFrame): The dataset
        columns (list of str or None): Columns to consider (defaults to all columns)
        threshold (float): Threshold in percentage (e.g., 5 means 5%)

    Returns:
        pd.DataFrame: Cleaned DataFrame with selected rows dropped
    """
    data = data.copy()

    if columns is None:
        columns = data.columns

    for col in columns:
        if col in data.columns:
            missing_pct = data[col].isnull().mean()
            if missing_pct < (threshold / 100):
                data = data[data[col].notnull()]

    return data


def remove_columns(data, columns=None, threshold=30):
    """
    Drop columns (specified or all) that have missing values
    exceeding a given threshold percentage.

    Parameters:
        data (pd.DataFrame): The dataset
        columns (list of str or None): Columns to consider (defaults to all columns)
        threshold (float): Threshold in percentage (e.g., 30 means 30%)

    Returns:
        pd.DataFrame: DataFrame with selected columns dropped
    """
    data = data.copy()

    if columns is None:
        columns = data.columns

    for col in columns:
        if col in data.columns:
            missing_pct = data[col].isnull().mean()
            if missing_pct > (threshold / 100):
                data = data.drop(columns=[col])

    return data


# Important imports for imputation to work!

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np


def impute_values(data, columns=None, strategy='auto', fill_value=None):
    """
    Impute missing values in a DataFrame using automatic or user-defined strategies.
    It's recommended to encode categorical columns when using strategies like MICE.

    Parameters:
        data (pd.DataFrame):
            The input DataFrame containing missing values.

        columns (list of str, optional):
            List of specific columns to impute.
            If None, all columns are considered.

        strategy (str):
            The imputation method to use. Supported values:
            - 'auto' (default): Chooses based on column type and missingness.
                - Numeric (<=5% missing): mean
                - Numeric (>5% missing): median
                - Categorical: mode
            - 'mean': Fill numeric columns with the column mean.
            - 'median': Fill numeric columns with the column median.
            - 'mode': Fill with the most frequent (mode) value.
            - 'constant': Fill with a user-defined value (`fill_value`).
            - 'ffill': Forward-fill from previous row.
            - 'bfill': Backward-fill from next row.
            - 'mice': Uses IterativeImputer (MICE-style multivariate imputation).

        fill_value (any, optional):
            Value used when `strategy='constant'`.

    Returns:
        pd.DataFrame:
            A copy of the original DataFrame with missing values imputed.

    Future Plans:
        Plan to support KNN, linear regression-based imputation, and categorical encoders.
    """
    data = data.copy()

    if columns is None:
        columns = data.columns

    if strategy == 'mice':
        # Only works with numeric columns
        numeric_cols = data[columns].select_dtypes(include=[np.number]).columns
        imputer = IterativeImputer(random_state=0)
        data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
        return data

    for col in columns:
        if col not in data.columns:
            continue

        col_dtype = data[col].dtype
        missing_pct = data[col].isnull().mean()

        if missing_pct == 0:
            continue

        if strategy == 'auto':
            if col_dtype in ['float64', 'int64']:
                if missing_pct <= 0.05:
                    data[col] = data[col].fillna(data[col].mean())
                else:
                    data[col] = data[col].fillna(data[col].median())
            else:
                mode = data[col].mode()
                if not mode.empty:
                    data[col] = data[col].fillna(mode[0])

        elif strategy == 'mean' and col_dtype in ['float64', 'int64']:
            data[col] = data[col].fillna(data[col].mean())

        elif strategy == 'median' and col_dtype in ['float64', 'int64']:
            data[col] = data[col].fillna(data[col].median())

        elif strategy == 'mode':
            mode = data[col].mode()
            if not mode.empty:
                data[col] = data[col].fillna(mode[0])

        elif strategy == 'constant':
            data[col] = data[col].fillna(fill_value)

        elif strategy in ['ffill', 'bfill']:
            data[col] = data[col].fillna(method=strategy)

    return data