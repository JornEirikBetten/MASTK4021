import pandas as pd 
import os 
import numpy as np 
import matplotlib.pyplot as plt 



path = os.getcwd()
fig_path = path + "/MA/Figures/"

data_path = path + "/MA/Data/bjornholt.csv"
df = pd.read_csv(data_path)
print(df)


fig = plt.figure()
plt.scatter(df["year"], df["days"], label = "Skiing days", color = "black", marker = '.')
plt.xlabel("year"); plt.ylabel("snow days")

plt.savefig(fig_path + "scatter_bjornholt.pdf", format = "pdf", bbox_inches = "tight")

