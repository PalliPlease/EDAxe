from EDAxe.preprocessing import *
from EDAxe.summary import *
import pandas as pd

#test for missing
# try:
data = pd.read_csv(r"D:\Downloads\202112-divvy-tripdata.csv\202112-divvy-tripdata.csv")
#     # Your dataset here to test
#     print(impute_values(data).isnull().sum())
# except:
#     print("Couldn't find the dataset!")

#test for summary
# data = pd.read_csv("https://raw.githubusercontent.com/atharvayeola/superstore-analytics-pipeline/master/superstore.csv")

#DONT FORGET TO USE nbformat BEFORE RUNNING THIS!
generate_summary(data)