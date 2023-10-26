import numpy as np 
import scipy.stats as sstats 
import matplotlib.pyplot as plt 
import os  
import pandas as pd 
from collections import Counter

rng = np.random.default_rng(10123)


def distribution(values, alpha): 
    n = len(values)
    if (n == 0): 
        value = rng.beta(1, 2)
        
    else: 
        number = rng.random()
        for i in range(len(values)): 
            if number < (i+1)/n*1/(alpha + 1): 
                value = values[i]
                break 
            else: 
                value = rng.beta(1, 2)
                
    return value 

alpha = 1
n = 1000 
values = []
for i in range(n): 
    value = distribution(values, alpha)
    values.append(value)
    
items = Counter(values).keys()
print("Number of unique outputs when alpha = 1: ", len(items))

alpha = 10
n = 1000 
values = []
for i in range(n): 
    value = distribution(values, alpha)
    values.append(value)
    
items = Counter(values).keys()
print("Number of unique outputs when alpha = 10: ", len(items))

alpha = 100
n = 1000 
values = []
for i in range(n): 
    value = distribution(values, alpha)
    values.append(value)
    
items = Counter(values).keys()
print("Number of unique outputs when alpha = 100: ", len(items))