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

#Edit data within file.

#Open file and set to a certain variable
df = pd.read_csv('thesavedones.csv', header=None)

#Nose locations
data = df.loc[::17,3]

reorg = pd.DataFrame(data, columns = ['Nose'])


# Generate plots of keypoint coordinate position vs time.