# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 18:10:18 2019

@author: Jaylen James
Code will plot the keypoint coordinates vs time in order to assign the maximum 
    value from this plot to the real-world distance measurement. This will be
    the label.
    
Meeting: 
    
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Edit data within file.

#Open file and set to a certain variable
ankle_df = pd.read_csv('jointTracker (16).csv', header=None) #This file has me pretty clearly tracked

plt.figure(1)
plt.plot(ankle_df[0])   #X - pixel position values\

plt.figure(2)
plt.plot(ankle_df[1])   #Y - pixel position values

print(ankle_df[0].min())

x_vals_df = ankle_df[0]
y_vals_df = ankle_df[1]


#Idea 3 - sorting
orderx_asce = x_vals_df.sort_values()
orderx_desc = x_vals_df.sort_values(ascending=False)

orderx_asce_df = orderx_asce.to_frame()     #Convert series to DataFrames
orderx_desc_df = orderx_desc.to_frame()


diffx_val = orderx_desc_df.subtract(orderx_asce_df.iloc[0,:], axis = 1) #Take difference of min and max values

#Assign index values to variables
descx_index_vals = orderx_desc_df.index.values.astype(int)[:]   
ascex_index_vals = orderx_asce_df.index.values.astype(int)[:]

#Take difference of index vals
diffx_index = np.subtract(descx_index_vals, ascex_index_vals)


#Repeat above process for y-values
ordery_asce = y_vals_df.sort_values()
ordery_desc = y_vals_df.sort_values(ascending=False)


# Generate plots of keypoint coordinate position vs time. Purpose is to observe
#   for peaks and smoothness.



