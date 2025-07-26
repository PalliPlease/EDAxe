"""
    CAUTION:
        DON'T FORGET TO USE nbformat, this renders it for jupyter style notebooks, if used in normal .py, it will
        throw an error!
    Description:
        Use this to get a quick overview of the data by simply calling a function.

    Future Plans;
        -> NEED TO ADD SUPPORT FOR .py!!!
        -> Support for handling high cardinality data (using dimensionality reduction or encodings?)
        -> Check for class imbalances
        -> More data types detection, right now the ones that aren't there are just directly being inserted into
           Categorical! OH NOES!
        -> Export to HTML as well
        -> Optimization?
        -> Hardcoded dtypes will fail
        -> Skew calc on non-numeric gives NaN
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
from scipy.stats import skew
from IPython.display import display, Markdown

warnings.filterwarnings("ignore")


def generate_summary(data, return_data=True, target=None, show=True, save=False, save_path=None):
    summary = {}

    # 1. Shape - Get the basic info about the data
    n_rows, n_cols = data.shape
    duplis = data.duplicated().sum()
    complete_rows = data.dropna().shape[0]

    summary['shape'] = {
        'rows': n_rows,
        'columns': n_cols,
        'duplicates': duplis,
        'complete_rows_percent': round(100 * complete_rows / n_rows, 2)
    }

    #2. Type - Get info on data type
    types = {
        'numeric': [],
        'categorical': [],
        'datetime': [],
        'constant': [],
        'high_cardinality': [],
        'id_like': []
    }

    for col in data.columns:
        # I'm using this to check the num of unique vals to help decide whether it's constant, high_card or ID_like
        # and also to classify the data types
        nunique = data[col].nunique()
        if data[col].dtype in ['int64', 'float64']:
            types['numeric'].append(col)
        elif data[col].dtype.kind in {'M', 'm'}:
            types['datetime'].append(col)
        else:
            types['categorical'].append(col)

        if nunique == 1:
            types['constant'].append(col)
        elif nunique > 0.5 * n_rows:  # I'm just taking a threshold of 50% for now
            types['high_cardinality'].append(col)
        if col.lower() in ['id', 'user_id', 'uid'] or nunique == n_rows:
            types['id_like'].append(col)

    summary['types'] = types

    # 3. Missing vals
    missing_per = data.isnull().mean() * 100
    summary['missing'] = {
        'columns_missing_over_50per': missing_per[missing_per > 50].index.tolist(),
        'columns_missing_over_10per': missing_per[missing_per > 10].index.tolist()
    }

    # 4. Target Summary - If a target is provided
    target_info = None
    if target and target in data.columns:
        # If numeric
        if data[target].dtype in ['int64', 'float64']:
            sk = round(skew(data[target].dropna()), 2)
            target_info = {
                'type': 'numeric',
                'skewness': sk,
                'missing_percent': round(data[target].isnull().mean() * 100, 2)
            }
        else:
            class_counts = data[target].value_counts(normalize=True)
            imbalance = class_counts.max() > 0.8
            target_info = {
                'type': 'categorical',
                'classes': class_counts.to_dict(),
                'imbalance': imbalance,  # Tells us about the imbalance but once again thresholded to above 80%
                'missing_percent': round(data[target].isnull().mean() * 100, 2)
            }
    summary['target'] = target_info

    # 5. Column Summary
    col_summary = []
    outlier_info = {}
    for col in data.columns:
        coltype = (
            'numeric' if col in types['numeric'] else
            'categorical' if col in types['categorical'] else
            'datetime' if col in types['datetime'] else
            'unknown'
        )
        outliers = None
        if coltype == 'numeric':  # If it's numeric then we are getting the outliers using IQR
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = data[(data[col] < lower) | (data[col] > upper)].index.tolist()
            if outliers:
                outlier_info[col] = outliers

        col_summary.append({
            'Column': col,
            'Type': coltype,
            'Unique': data[col].nunique(),
            'Missing %': round(data[col].isnull().mean() * 100, 2),
            'Skew': round(skew(data[col].dropna()), 2) if coltype == 'numeric' else None,
            'Constant': 'CONST' if col in types['constant'] else '',
            'Example Values': data[col].dropna().astype(str).unique()[:3].tolist()
        })
    summary['column_summary_df'] = pd.DataFrame(col_summary)
    summary['outliers'] = outlier_info

    # 6. Warnings - Simple warnings from all the data we collected hehe
    warnings_list = []
    if duplis > 0:
        warnings_list.append(f"{duplis} duplicate rows found.")
    for col in summary['missing']['columns_missing_over_50per']:
        warnings_list.append(f"'{col}' has over 50% missing values.")  # Should I also show missing above 10%?
    for col in types['constant']:
        warnings_list.append(f"'{col}' is constant and can be dropped.")
    for col in types['id_like']:
        warnings_list.append(f"'{col}' looks like an ID column.")
    if outlier_info:
        for col, idxs in outlier_info.items():
            warnings_list.append(f"'{col}' has {len(idxs)} outliers.")
    summary['warnings'] = warnings_list

    # 7. Recommendations - Need to add more reccs
    recommendations = []
    if target_info:
        if target_info.get('type') == 'numeric' and abs(target_info.get('skewness', 0)) > 2:
            recommendations.append(f"Consider log-transforming '{target}' due to high skew.")
        if target_info.get('type') == 'categorical' and target_info.get('imbalance'):
            recommendations.append(f"Target '{target}' is imbalanced. Consider stratified sampling or balancing.")
    summary['recommendations'] = recommendations

    # 8. Data Readiness Score - Now this is something genius :)
    score = 10
    score -= len(summary['missing']['columns_missing_over_50per']) * 1.5  # Missing data very bad
    score -= len(types['constant']) * 0.5
    score -= len(types['id_like']) * 1
    score -= 1 if target_info and target_info.get('type') == 'numeric' and abs(
        target_info.get('skewness', 0)) > 2 else 0
    score = max(0, round(score, 1))
    summary['ml_readiness_score'] = score

    # Displaying the info at last
    if show:
        display(Markdown(f"### Dataset Summary ({n_rows} rows × {n_cols} columns)"))
        print("\n".join(summary['warnings']))
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"-> {rec}")
        print(f"\nDataReadiness Score: {score} / 10")

        # pIE cHART
        type_counts = {k: len(v) for k, v in types.items() if k in ['numeric', 'categorical', 'datetime']}
        fig = px.pie(names=list(type_counts.keys()), values=list(type_counts.values()), title="Feature Types")
        fig.show()

        # Missing Bar Chart
        top_missing = missing_per[missing_per > 0].sort_values(ascending=False).head(10)
        if not top_missing.empty:
            sns.barplot(x=top_missing.values, y=top_missing.index, palette="hot")
            plt.title("Top Missing Columns")
            plt.xlabel("Missing %")
            plt.show()

        # Target Distribution
        if target_info:
            if target_info['type'] == 'numeric':
                sns.histplot(data[target], kde=True, color='skyblue')
                plt.title(f"Distribution of Target: {target}")
                plt.show()
            elif target_info['type'] == 'categorical':
                sns.countplot(x=data[target], palette='viridis')
                plt.title(f"Class Balance: {target}")
                plt.show()

        # Correlation Heatmap
        if len(types['numeric']) > 1:
            corr = data[types['numeric']].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Correlation Matrix (Numeric Features)")
            plt.show()

        # Time trend plots
        if types['datetime'] and target:
            for dt_col in types['datetime']:
                try:
                    temp = data[[dt_col, target]].dropna().sort_values(dt_col)
                    if target_info['type'] == 'numeric':
                        sns.lineplot(x=temp[dt_col], y=temp[target])
                        plt.title(f"Trend of '{target}' over '{dt_col}'")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plt.show()
                    elif target_info['type'] == 'categorical':
                        temp['count'] = 1
                        temp = temp.groupby([dt_col, target]).count().reset_index()
                        sns.lineplot(data=temp, x=dt_col, y='count', hue=target)
                        plt.title(f"Category Count of '{target}' over '{dt_col}'")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plt.show()
                except Exception:
                    continue

        display(summary['column_summary_df'])

    if save and save_path:
        summary['column_summary_df'].to_csv(save_path, index=False)

    if return_data:
        return summary
