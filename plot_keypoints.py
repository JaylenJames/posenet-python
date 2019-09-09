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
part_data_df= df.iloc[::17,3]

#Remove brackets and spaces before string and after numbers.
part_edit_df = part_data_df.map(lambda x: x.lstrip('[ ').rstrip(' ]'))

#Replace spaces between numbers with a comma
part_edit2_df = part_edit_df.map(lambda x: x.replace("   ",",").replace("  ",",").replace(" ",","))

#splits string into muliple columns
part_split_df = part_edit2_df.str.split(",",expand=True)


#convert keypoint coordinates to integer values
#part_data_ints_df = part_split_df["0"].astype(int) 
part_data_ints_col0_df = pd.to_numeric(part_split_df[0])
part_data_ints_col1_df = pd.to_numeric(part_split_df[1])

#Concatenate columns
part_location_data = pd.concat([part_data_ints_col0_df, part_data_ints_col1_df], axis=1)


reorg = pd.DataFrame(part_data_df, columns = ['Nose'])


# Generate plots of keypoint coordinate position vs time.