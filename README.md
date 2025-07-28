# EDAxe - Cut Through Data 

- EDAxe is a Python library for automatic exploratory data analysis (EDA), missing‑value handling, and data cleaning.
---

## Key Features

- **Missing value imputation**: mean, median, mode, MICE, etc.  
- **Filtering tools**: drop rows/columns by missing‑value thresholds  
- **Summary statistics**: automatic feature‑type detection, outlier analysis, skewness, missingness metrics, and ML readiness score  
- **Interactive visualizations**: pie charts, bar charts, heatmaps, trend plotting (Plotly + Seaborn)  (COMING SOON)
- **Saving support**: export summary tables to CSV  

---

## Installation

```bash
pip install git+https://github.com/PalliPlease/EDAxe.git
```

This installs **EDAxe** and its dependencies into your active Python environment.

---

##  Usage & Setup Guide (for Full Visuals in Jupyter / VSCode)

### 1. Create and activate a virtual environment (recommended)
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install EDAxe and its dependencies
```bash
pip install git+https://github.com/PalliPlease/EDAxe.git
pip install pandas seaborn matplotlib plotly scipy jupyter ipykernel
```

### 3. Register the `.venv` as a Jupyter / VSCode kernel
```bash
python -m ipykernel install --user --name=edaxe-env --display-name "Python (.venv - EDAxe)"
```
→ Now choose the **Python (.venv - EDAxe)** kernel when running notebooks.

### 4. Set Plotly renderer for interactive charts
Add to your notebook or at the start of your script:
```python
import plotly.io as pio
pio.renderers.default = "notebook"       # Jupyter
# or use "vscode" or "iframe" for other environments
```

---

##  Example Usage

```python
import pandas as pd
import plotly.io as pio

pio.renderers.default = "notebook"

from EDAxe.summary import generate_summary

df = pd.read_csv("your_data.csv")
summary = generate_summary(df, target="your_target_column", save=True, save_path="eda_summary.csv")
```

- A **pie chart of feature types** will be shown in notebook
- **Missing value bar plot**, **correlation heatmap**, and **trend line plots** (if datetime & target present)
- You can find the **column summary table** returned as a `pd.DataFrame` and saved if requested

---

## Troubleshooting Table

| Issue | Cause & Fix |
|-------|-------------|
| Pie chart doesn't display | Add `pio.renderers.default = "notebook"` before running `generate_summary()` |
| Import fails inside notebook | The notebook may not be using your `.venv`; switch to the registered **Python (.venv - EDAxe)** kernel |
| Only one categorical feature appears in pie chart | Validate that Plotly renderer is configured correctly and that your dataset has proper numeric/datetime types |
| Modifications to EDAxe not reflected | Restart the kernel or reinstall using `pip install -e .` in project root inside `.venv` |

---

## Quick Data Demo

```python
import plotly.io as pio
pio.renderers.default = "notebook"

import pandas as pd
from EDAxe.summary import generate_summary

df = pd.DataFrame({
  'age': [25, 30, 30, 40, None],
  'signup_date': pd.date_range("2021-01-01", periods=5),
  'gender': ['M', 'F', 'F', 'M', 'F']
})

generate_summary(df, target="age")
```

---

## Contributing & Development

- Clone this repo and work inside `.venv`
- Install in editable mode: `pip install -e .`  
- Write and test notebooks using the registered `.venv` kernel  
- Propose enhancements: HTML export, improved type detection, encoding for high-cardinality, etc.
