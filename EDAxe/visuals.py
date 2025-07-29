"""
Description:
    Functions to handle data visualization. These "macros" are designed to
    generate insightful plots with minimal code. They can accept either a
    pandas DataFrame or a path to a CSV file as input.

Current Functions:
    Total: 8
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

def _load_data(data):
    """Helper function to load data if it's a file path."""
    if isinstance(data, str):
        try:
            return pd.read_csv(data)
        except FileNotFoundError:
            warnings.warn(f"File not found at path: {data}. Skipping operation.")
            return None
        except Exception as e:
            warnings.warn(f"Error reading file '{data}': {e}. Skipping operation.")
            return None
    elif isinstance(data, pd.DataFrame):
        return data.copy()
    else:
        raise TypeError("Input 'data' must be a pandas DataFrame or a file path (str).")

def plot_distribution(data, column, title=None, xlabel=None, save_path=None):
    """
    Plots the distribution of a single numerical column.
    Accepts a DataFrame or a CSV file path.
    """
    df = _load_data(data)
    if df is None or column not in df.columns:
        warnings.warn(f"Column '{column}' not found or data could not be loaded. Skipping plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(title or f'Distribution of {column}', fontsize=16)
    plt.xlabel(xlabel or column, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_countplot(data, column, title=None, xlabel=None, save_path=None):
    """
    Generates a count plot for a single categorical column.
    Accepts a DataFrame or a CSV file path.
    """
    df = _load_data(data)
    if df is None or column not in df.columns:
        warnings.warn(f"Column '{column}' not found or data could not be loaded. Skipping plot.")
        return

    plt.figure(figsize=(12, 7))
    sns.countplot(y=df[column], order=df[column].value_counts().index)
    plt.title(title or f'Count of {column}', fontsize=16)
    plt.xlabel(xlabel or 'Count', fontsize=12)
    plt.ylabel(column, fontsize=12)

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_barchart(data, x_col, y_col, title=None, xlabel=None, ylabel=None, save_path=None):
    """
    Creates a bar chart between a categorical and a numerical variable.
    Accepts a DataFrame or a CSV file path.
    """
    df = _load_data(data)
    if df is None or x_col not in df.columns or y_col not in df.columns:
        warnings.warn(f"One or more columns not found or data could not be loaded. Skipping plot.")
        return

    plt.figure(figsize=(12, 7))
    sns.barplot(x=df[x_col], y=df[y_col])
    plt.title(title or f'{y_col} by {x_col}', fontsize=16)
    plt.xlabel(xlabel or x_col, fontsize=12)
    plt.ylabel(ylabel or y_col, fontsize=12)
    plt.xticks(rotation=45)

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_scatterplot(data, x_col, y_col, hue=None, title=None, xlabel=None, ylabel=None, save_path=None):
    """
    Generates a scatter plot between two numerical variables.
    Accepts a DataFrame or a CSV file path.
    """
    df = _load_data(data)
    if df is None or x_col not in df.columns or y_col not in df.columns:
        warnings.warn(f"One or more columns not found or data could not be loaded. Skipping plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue)
    plt.title(title or f'Scatter Plot of {y_col} vs. {x_col}', fontsize=16)
    plt.xlabel(xlabel or x_col, fontsize=12)
    plt.ylabel(ylabel or y_col, fontsize=12)

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_linechart(data, x_col, y_col, hue=None, title=None, xlabel=None, ylabel=None, save_path=None):
    """
    Creates a line chart, ideal for time-series data.
    Accepts a DataFrame or a CSV file path.
    """
    df = _load_data(data)
    if df is None or x_col not in df.columns or y_col not in df.columns:
        warnings.warn(f"One or more columns not found or data could not be loaded. Skipping plot.")
        return
        
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=x_col, y=y_col, hue=hue)
    plt.title(title or f'Line Chart of {y_col} over {x_col}', fontsize=16)
    plt.xlabel(xlabel or x_col, fontsize=12)
    plt.ylabel(ylabel or y_col, fontsize=12)

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_boxplot(data, x_col, y_col, title=None, xlabel=None, ylabel=None, save_path=None):
    """
    Generates a box plot to show distributions across categories.
    Accepts a DataFrame or a CSV file path.
    """
    df = _load_data(data)
    if df is None or x_col not in df.columns or y_col not in df.columns:
        warnings.warn(f"One or more columns not found or data could not be loaded. Skipping plot.")
        return

    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x=x_col, y=y_col)
    plt.title(title or f'Box Plot of {y_col} by {x_col}', fontsize=16)
    plt.xlabel(xlabel or x_col, fontsize=12)
    plt.ylabel(ylabel or y_col, fontsize=12)
    plt.xticks(rotation=45)

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_correlation_heatmap(data, title='Correlation Matrix', cmap='coolwarm', save_path=None):
    """
    Plots the correlation matrix of numerical columns.
    Accepts a DataFrame or a CSV file path.
    """
    df = _load_data(data)
    if df is None:
        warnings.warn("Data could not be loaded. Skipping plot.")
        return
        
    numeric_data = df.select_dtypes(include='number')
    if numeric_data.shape[1] < 2:
        warnings.warn("Not enough numeric columns to create a correlation heatmap.")
        return

    corr = numeric_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, linewidths=.5)
    plt.title(title, fontsize=16)

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_pairplot(data, hue=None, vars=None, save_path=None):
    """
    Creates a grid of scatterplots for pairwise relationships.
    Accepts a DataFrame or a CSV file path.
    """
    df = _load_data(data)
    if df is None:
        warnings.warn("Data could not be loaded. Skipping plot.")
        return

    if vars is None:
        plot_vars = df.select_dtypes(include='number').columns
    else:
        plot_vars = vars

    if len(plot_vars) < 2:
        warnings.warn("Not enough numeric columns for a pair plot.")
        return

    pair_plot = sns.pairplot(df, vars=plot_vars, hue=hue)
    pair_plot.fig.suptitle('Pairwise Relationships', y=1.02, fontsize=16)

    if save_path:
        pair_plot.savefig(save_path)
    plt.show()
