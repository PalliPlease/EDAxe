from EDAxe.missing import *
import pandas as pd

try:
    data = pd.read_csv(r"D:\Downloads\202112-divvy-tripdata.csv\202112-divvy-tripdata.csv")
    # Your dataset here to test
    print(impute_values(data).isnull().sum())
except:
    print("Couldn't find the dataset!")