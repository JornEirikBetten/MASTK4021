import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import math 
import os 

path = os.getcwd()
fig_path = path + "/MA/Figures/"

def find_quantile(quantile, distribution, original_variable): 
    normalized_distribution = distribution/np.sum(distribution)
    quantile_value = 0 
    idx = int(0)
    while quantile_value < quantile: 
        quantile_value += normalized_distribution[idx]
        idx += int(1) 
        
    quantile_value_variable = original_variable[idx-1]
    returns = [quantile_value_variable, quantile_value, idx-1]
    return returns 


def predictive_distribution(a, b, data, new_data_point): 
    square_root_data = np.sqrt(data)
    return (b + np.sum(square_root_data))**(a+len(data))*(a + len(data))/(b + np.sum(square_root_data) + np.sqrt(new_data_point))**(a + len(data) + 1)


data = np.array([0.771, 0.140, 0.135, 0.007, 0.088, 0.008,
        0.268, 0.022, 0.131, 0.142, 0.421, 0.125])

a = 4.4 
b = 2.2 

y = np.linspace(0, 100, 1000001)
pred_density = np.array([predictive_distribution(a, b, data, data_point) for data_point in y])
print(np.sum(pred_density))

quantile_010 = find_quantile(0.10, pred_density, y)
quantile_050 = find_quantile(0.50, pred_density, y)
quantile_090 = find_quantile(0.90, pred_density, y)

print("Quantile 0.10: ")
print(f"Y = {quantile_010[0]}, quantile_value = {quantile_010[1]}")

print("Quantile 0.50: ")
print(f"Y = {quantile_050[0]}, quantile_value = {quantile_050[1]}")

print("Quantile 0.90: ")
print(f"Y = {quantile_090[0]}, quantile_value = {quantile_090[1]}")

plt.plot(y, pred_density, label = "Predictive density", color = "black", linestyle = "solid")
plt.xlabel("value")
plt.ylabel("probability density")
plt.yticks([], [])
plt.legend()
plt.savefig(fig_path + "predictive_density.pdf", format = "pdf", bbox_inches = "tight")