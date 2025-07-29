"""
Description:
    Functions to handle preprocessing.

Current Functions:
    Total: 4

Future Plans:

"""


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
        Plan to support KNN, linear regression-based imputation, and categorical data.
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



import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder

class AutoEncodingPreprocessor:
    """
    AutoEncodingPreprocessor is a utility class to encode categorical variables in a DataFrame
    using various strategies: label encoding, ordinal encoding, or one-hot encoding.

    Features:
    - Supports automatic column detection or manual column selection
    - Allows encoding strategy selection per use case
    - Maintains consistency across train-test splits
    - Exposes category mappings for label/ordinal/one-hot encoded features

    Parameters:
    ----------
    strategy : str, default='auto'
        Encoding strategy to use. Options:
        - 'auto' : uses one-hot for low cardinality, ordinal for high
        - 'label' : uses sklearn LabelEncoder (for single columns)
        - 'ordinal' : uses sklearn OrdinalEncoder
        - 'onehot' : uses sklearn OneHotEncoder

    cardinality_threshold : int, default=10
        Threshold used in 'auto' mode to switch between one-hot and ordinal.

    columns : list or None, default=None
        Columns to encode. If None, all categorical columns are auto-detected.
    """

    def __init__(self, strategy='auto', cardinality_threshold=10, columns=None):
        self.strategy = strategy
        self.cardinality_threshold = cardinality_threshold
        self.columns = columns
        self.encoders = {}
        self.column_types = {}
        self.ohe_columns = []

    def fit(self, df):
        """
        Fit encoders to the specified or inferred categorical columns of a DataFrame.

        Parameters:
        ----------
        df : pandas.DataFrame
            The input DataFrame to fit the encoders on.
        """
        if self.columns is None:
            target_cols = df.select_dtypes(include=['object', 'category']).columns
        else:
            target_cols = self.columns

        for col in target_cols:
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found in DataFrame. Skipping.")
                continue

            unique_vals = df[col].nunique(dropna=True)

            if self.strategy == 'onehot' or (self.strategy == 'auto' and unique_vals <= self.cardinality_threshold):
                enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                enc.fit(df[[col]])
                self.encoders[col] = enc
                self.column_types[col] = 'onehot'

            elif self.strategy == 'ordinal':
                enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                enc.fit(df[[col]])
                self.encoders[col] = enc
                self.column_types[col] = 'ordinal'

            elif self.strategy == 'label':
                enc = LabelEncoder()
                enc.fit(df[col].astype(str))
                self.encoders[col] = enc
                self.column_types[col] = 'label'

            elif self.strategy == 'auto':
                enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                enc.fit(df[[col]])
                self.encoders[col] = enc
                self.column_types[col] = 'ordinal'

    def transform(self, df):
        """
        Transform a DataFrame using the previously fitted encoders.

        Parameters:
        ----------
        df : pandas.DataFrame
            The DataFrame to transform.

        Returns:
        -------
        pandas.DataFrame
            A DataFrame containing the encoded columns.
        """
        transformed = []

        for col in self.encoders:
            encoder = self.encoders[col]
            typ = self.column_types[col]

            if typ == 'onehot':
                encoded = encoder.transform(df[[col]])
                ohe_df = pd.DataFrame(
                    encoded,
                    columns=encoder.get_feature_names_out([col]),
                    index=df.index
                )
                self.ohe_columns.extend(ohe_df.columns)
                transformed.append(ohe_df)

            elif typ == 'ordinal':
                encoded = encoder.transform(df[[col]])
                transformed.append(pd.DataFrame(encoded, columns=[col], index=df.index))

            elif typ == 'label':
                encoded = encoder.transform(df[col].astype(str))
                transformed.append(pd.DataFrame(encoded, columns=[col], index=df.index))

        return pd.concat(transformed, axis=1)

    def fit_transform(self, df):
        """
        Fit the encoders and transform the input DataFrame in a single step.

        Parameters:
        ----------
        df : pandas.DataFrame
            The input DataFrame to fit and transform.

        Returns:
        -------
        pandas.DataFrame
            The transformed DataFrame.
        """
        self.fit(df)
        return self.transform(df)

    def get_mappings(self):
        """
        Retrieve the encoding mappings used by each encoder.

        Returns:
        -------
        dict
            A dictionary of the form:
            - For label/ordinal: {column: {encoded_value: category}}
            - For one-hot: {column: [category1, category2, ...]} showing the category order
        """
        mappings = {}
        for col, enc in self.encoders.items():
            typ = self.column_types[col]
            if typ == 'label':
                mappings[col] = {i: cat for i, cat in enumerate(enc.classes_)}
            elif typ == 'ordinal':
                mappings[col] = {i: cat for i, cat in enumerate(enc.categories_[0])}
            elif typ == 'onehot':
                mappings[col] = list(enc.categories_[0])
        return mappings


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class AutoScalerPreprocessor:
    """
    Automatically scales numeric features using a chosen or inferred strategy.

    Supported Scalers:
    - StandardScaler (mean=0, std=1)
    - MinMaxScaler (scales to 0–1)
    - RobustScaler (uses IQR, robust to outliers)

    Parameters
    ----------
    strategy : str, default='auto'
        One of {'auto', 'standard', 'minmax', 'robust'}.
        If 'auto', strategy is chosen per column using heuristics.

    columns : list or None, default=None
        Columns to scale. If None, all numeric columns are auto-detected.
    """

    def __init__(self, strategy='auto', columns=None):
        self.strategy = strategy
        self.columns = columns
        self.scalers = {}
        self.column_strategies = {}

    def _choose_strategy(self, series):
        """
        Heuristics to choose the best scaler for a column:
        - MinMaxScaler if range is small
        - StandardScaler if low skew and std is reasonable
        - RobustScaler if high skew or wide outliers
        - Raise error if no clear choice
        """
        std = series.std()
        min_, max_ = series.min(), series.max()
        range_ = max_ - min_
        skew = series.skew()

        if range_ <= 10:
            return 'minmax'
        elif abs(skew) < 1 and std < 1000:
            return 'standard'
        elif abs(skew) >= 1 or range_ > 1000:
            return 'robust'
        else:
            raise ValueError(
                f"Unable to determine scaling strategy for column '{series.name}'. "
                f"Please specify manually using strategy='standard'|'minmax'|'robust'."
            )

    def fit(self, df):
        """
        Fit scalers for selected numeric columns.

        Parameters
        ----------
        df : pandas.DataFrame
            The training DataFrame.
        """
        if self.columns is None:
            self.columns = df.select_dtypes(include='number').columns.tolist()

        for col in self.columns:
            series = df[col]
            strat = self.strategy

            if strat == 'auto':
                strat = self._choose_strategy(series)

            if strat == 'standard':
                scaler = StandardScaler()
            elif strat == 'minmax':
                scaler = MinMaxScaler()
            elif strat == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unsupported scaling strategy: {strat}")

            self.scalers[col] = scaler.fit(series.values.reshape(-1, 1))
            self.column_strategies[col] = strat

    def transform(self, df):
        """
        Transform the DataFrame using previously fitted scalers.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
            Scaled numeric DataFrame.
        """
        scaled_cols = []
        for col in self.columns:
            scaler = self.scalers[col]
            series = df[col].values.reshape(-1, 1)
            scaled = scaler.transform(series)
            scaled_cols.append(pd.DataFrame(scaled, columns=[col], index=df.index))

        return pd.concat(scaled_cols, axis=1)

    def inverse_transform(self, df):
        """
        Undo scaling for scaled columns.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
            Reconstructed original scale DataFrame.
        """
        unscaled_cols = []
        for col in self.columns:
            scaler = self.scalers[col]
            series = df[col].values.reshape(-1, 1)
            unscaled = scaler.inverse_transform(series)
            unscaled_cols.append(pd.DataFrame(unscaled, columns=[col], index=df.index))

        return pd.concat(unscaled_cols, axis=1)

    def fit_transform(self, df):
        """
        Fit and transform the DataFrame in one step.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
            Scaled DataFrame.
        """
        self.fit(df)
        return self.transform(df)

    def get_scaler_info(self):
        """
        Get a summary of which scaler was applied to which column.

        Returns
        -------
        dict: {column_name: strategy_used}
        """
        return self.column_strategies

