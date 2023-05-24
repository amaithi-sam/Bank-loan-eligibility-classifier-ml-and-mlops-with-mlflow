import sys 
import os 
import numpy as np 
import pandas as pd 




def drop_missing_values(data):
    data = data.dropna()
    print("After deleting missing values")
    visualize_missing_values(data)
    return data 

df1 = drop_missing_values(data)

    

