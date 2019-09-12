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

#Edit data within file.

#Open file and set to a certain variable
df = pd.read_csv('thesavedones.csv', header=None)

#Nose locations
part_data_df= df.iloc[::17,3]
rankle_data_df = df.iloc[::13,3]

    #Create loop to perform this data manipulation for each data set, starting here:
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
    #And ending here.

reorg = pd.DataFrame(part_location_data, columns = ['Nose-x', 'Nose-y'])


# Generate plots of keypoint coordinate position vs time.

plt.plot(part_location_data[0])
plt.show()

