"""
    Description:
        Use this to get a quick overview of the data by simply calling a function.

    Future Plans;
        -> IDK blud, just give me some time
        -> Support for handling high cardinality data (using dimensionality reduction or encodings?)
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import skew

warnings.filterwarnings("ignore")


def generate_summary(data, target=None):
    summary = {}

    # 1. Shape - Get the basic info about the data
    n_rows, n_cols = data.shape
    duplis = data.duplicated.sum()
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