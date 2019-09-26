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

#Keypoint locations
nose_data_df= df.iloc[::17,3]
rankle_data_df = df.iloc[16::17,3]
rknee_data_df = df.iloc[14::17,3]
rwrist_data_df = df.iloc[10::17,3]
reye_data_df = df.iloc[2::17,3]

data_sets = [nose_data_df, rankle_data_df, rknee_data_df, rwrist_data_df, reye_data_df]
kps = ["Nose","Right Ankle","Right Knee","Right Wrist", "Right Eye"]

locations = [] #Initializing a list in order to append df at end of loop

#Create loop to perform this data manipulation for each data set:
for i in range(len(data_sets)):
    #Remove brackets and spaces before string and after numbers.
    data_select = data_sets[i]
    
    part_edit_df = data_select.map(lambda x: x.lstrip('[ ').rstrip(' ]'))
    
    #Replace spaces between numbers with a comma
    part_edit2_df = part_edit_df.map(lambda x: x.replace("   ",",").replace("  ",",").replace(" ",","))
    
    #splits string into muliple columns
    part_split_df = part_edit2_df.str.split(",",expand=True)
    
    #convert keypoint coordinates to integer values 
    part_data_ints_col0_df = pd.to_numeric(part_split_df[0])
    part_data_ints_col1_df = pd.to_numeric(part_split_df[1])
    
    #Concatenate columns
    part_location_data = pd.concat([part_data_ints_col0_df, part_data_ints_col1_df], axis=1)
    
    locations.append(part_location_data) #saves the location data to the locations variable
    
    
# Rename row elements of locations list to match appropriate keypoint names


# Generate plots of keypoint coordinate position vs time. Purpose is to observe
#   for peaks and smoothness.
for i in range(len(data_sets)):
    plt.figure(i, figsize=(13, 5))
    
    plt.subplot(121)
    plt.plot(locations[i][0])
    plt.subplot(122)
    plt.plot(locations[i][1], 'm') #y-values for right wrist
    plt.suptitle(kps[i] + ' X vs Frame and Y vs Frame Coordinates')
    plt.show() 










