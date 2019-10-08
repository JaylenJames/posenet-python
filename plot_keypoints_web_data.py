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
from scipy.signal import find_peaks
from scipy.signal import medfilt
from scipy.signal import peak_prominences
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

#Edit data within file.

#Open file and set to a certain variable
ankle_df = pd.read_csv('jointTracker (16).csv', header=None) #This file has me pretty clearly tracked

rknee1_df = pd.read_csv('20191002_rknee_pos_1st.csv', header=None)
rknee2_df = pd.read_csv('20191002_rknee_pos_2nd.csv', header=None)
rknee3_df = pd.read_csv('20191002_rknee_pos_3rd.csv', header=None)
rknee4_df = pd.read_csv('20191002_rknee_pos_4th.csv', header=None)
rknee5_df = pd.read_csv('20191002_rknee_pos_5th.csv', header=None)
rknee6_df = pd.read_csv('20191002_rknee_pos_6th.csv', header=None)
rknee7_df = pd.read_csv('20191002_rknee_pos_7th.csv', header=None)
rknee8_df = pd.read_csv('20191002_rknee_pos_8th.csv', header=None)

real_measures = np.array([32,33,32,35,35,
                          32,32,32,33,35,
                          34,36,35,35,34,34,35,35,34,35,
                          31,33,37,34,33,33,33,35,35,35,
                          30,31,33,23,25,28,28,29,31,42,
                          32,31.5,24,29,37,36,31,34,28,33.5,
                          38,38,42,42,42,41,43,38,39,40,
                          32,34,41,36,36,35,37,36,38,40]) #Document real measures
    
real_measures_df = pd.DataFrame(data=real_measures[0:]) #Convert to a DataFrame


#Tabulate height and weight columns
heights_df = pd.DataFrame({"Height": [69]*50 + [69.5]*10 + [67]*10}) #Heights in inches

weights_df = pd.DataFrame({"Weight": [165]*50 + [215]*10 + [160]*10}) #Weights in pounds


#Assign x and y position values to variables
x_rknee1 = rknee1_df[0]
y_rknee1 = rknee1_df[1]

x_rknee2 = rknee2_df[0]
y_rknee2 = rknee2_df[1]

x_rknee3 = rknee3_df[0]
y_rknee3 = rknee3_df[1]

x_rknee4 = rknee4_df[0]
y_rknee4 = rknee4_df[1]

x_rknee5 = rknee5_df[0]
y_rknee5 = rknee5_df[1]

x_rknee6 = rknee6_df[0]
y_rknee6 = rknee6_df[1]

x_rknee7 = rknee7_df[0]
y_rknee7 = rknee7_df[1]

x_rknee8 = rknee8_df[0]
y_rknee8 = rknee8_df[1]

################################Obtain X peak prominences######################
#Plot the values to view visually
#plt.figure(1)
#plt.plot(x_rknee1)   
#
#plt.figure(2)
#plt.plot(x_rknee2)   
#
#plt.figure(3)
#plt.plot(x_rknee3)
#
#plt.figure(4)
#plt.plot(x_rknee4)

#Apply smoothing with median filter
filterx_rk_1 = medfilt(x_rknee1, kernel_size = 13)
filterx_rk_2 = medfilt(x_rknee2, kernel_size = 13)
filterx_rk_3 = medfilt(x_rknee3, kernel_size = 13)
filterx_rk_4 = medfilt(x_rknee4, kernel_size = 13)
filterx_rk_5 = medfilt(x_rknee5, kernel_size = 13)
filterx_rk_6 = medfilt(x_rknee6, kernel_size = 13)
filterx_rk_7 = medfilt(x_rknee7, kernel_size = 13)
filterx_rk_8 = medfilt(x_rknee8, kernel_size = 13)

#Plot values to view smoothed plot visually
plt.figure(10)
plt.plot(filterx_rk_1)

plt.figure(11)
plt.plot(filterx_rk_2)

plt.figure(12)
plt.plot(filterx_rk_3)

plt.figure(13)
plt.plot(filterx_rk_4)

plt.figure(14)
plt.plot(filterx_rk_5)

plt.figure(15)
plt.plot(filterx_rk_6)

plt.figure(16)
plt.plot(filterx_rk_7)

plt.figure(17)
plt.plot(filterx_rk_8)

#Obtain peaks and prominences
peaksx_rk1, _ = find_peaks(filterx_rk_1, height=180)
promsx_1 = peak_prominences(filterx_rk_1, peaksx_rk1)
promsx_1_df = pd.DataFrame(data=promsx_1[0][0:]) #,    # values
#...              index=data[1:,0],    # 1st column as index
#...              columns=data[0,1:])  # 1st row as the column names
                                      # Convert to DataFrame

peaksx_rk2, _ = find_peaks(filterx_rk_2, height=200)
promsx_2 = peak_prominences(filterx_rk_2, peaksx_rk2)
promsx_2_df = pd.DataFrame(data=promsx_2[0][0:])

peaksx_rk3, _ = find_peaks(filterx_rk_3, height=220)
promsx_3 = peak_prominences(filterx_rk_3, peaksx_rk3)
promsx_3_df = pd.DataFrame(data=promsx_3[0][0:])

peaksx_rk4, _ = find_peaks(filterx_rk_4, height=180)
promsx_4 = peak_prominences(filterx_rk_4, peaksx_rk4)
promsx_4_df = pd.DataFrame(data=promsx_4[0][0:])

peaksx_rk5, _ = find_peaks(filterx_rk_5, height=230)
promsx_5 = peak_prominences(filterx_rk_5, peaksx_rk5)
promsx_5_df = pd.DataFrame(data=promsx_5[0][0:])

peaksx_rk6, _ = find_peaks(filterx_rk_6, height=200)
promsx_6 = peak_prominences(filterx_rk_6, peaksx_rk6)
promsx_6_df = pd.DataFrame(data=promsx_6[0][0:])

peaksx_rk7, _ = find_peaks(filterx_rk_7, height=200)
promsx_7 = peak_prominences(filterx_rk_7, peaksx_rk7)
promsx_7_df = pd.DataFrame(data=promsx_7[0][0:])

peaksx_rk8, _ = find_peaks(filterx_rk_8, height=150)
promsx_8 = peak_prominences(filterx_rk_8, peaksx_rk8)
promsx_8_df = pd.DataFrame(data=promsx_8[0][0:])




######################Obtian Y value peak prominences########################
#Plot the values to view visually
#plt.figure(20)
#plt.plot(y_rknee1, 'g')   
#
#plt.figure(21)
#plt.plot(y_rknee2, 'g')   
#
#plt.figure(23)
#plt.plot(y_rknee3, 'g')
#
#plt.figure(24)
#plt.plot(y_rknee4, 'g')


#Apply smoothing with median filter
filtery_rk_1 = medfilt(y_rknee1, kernel_size = 13)
filtery_rk_2 = medfilt(y_rknee2, kernel_size = 13)
filtery_rk_3 = medfilt(y_rknee3, kernel_size = 13)
filtery_rk_4 = medfilt(y_rknee4, kernel_size = 13)
filtery_rk_5 = medfilt(y_rknee5, kernel_size = 13)
filtery_rk_6 = medfilt(y_rknee6, kernel_size = 13)
filtery_rk_7 = medfilt(y_rknee7, kernel_size = 13)
filtery_rk_8 = medfilt(y_rknee8, kernel_size = 13)


#Plot values to view smoothed plot visually
plt.figure(30)
plt.plot(filtery_rk_1, 'g')

plt.figure(31)
plt.plot(filtery_rk_2, 'g')

plt.figure(32)
plt.plot(filtery_rk_3, 'g')

plt.figure(33)
plt.plot(filtery_rk_4, 'g')

plt.figure(34)
plt.plot(filtery_rk_5, 'g')

plt.figure(35)
plt.plot(filtery_rk_6, 'g')

plt.figure(36)
plt.plot(filtery_rk_7, 'g')

plt.figure(37)
plt.plot(filtery_rk_8, 'g')


#Obtain peaks and prominences
peaksy_rk1, _ = find_peaks(filtery_rk_1, height=340)
promsy_1 = peak_prominences(filtery_rk_1, peaksy_rk1)
promsy_1_df = pd.DataFrame(data=promsy_1[0][0:])

peaksy_rk2, _ = find_peaks(filtery_rk_2, height=312)
promsy_2 = peak_prominences(filtery_rk_2, peaksy_rk2)
promsy_2_df = pd.DataFrame(data=promsy_2[0][0:])

peaksy_rk3, _ = find_peaks(filtery_rk_3, height=321)
promsy_3 = peak_prominences(filtery_rk_3, peaksy_rk3)
promsy_3_df = pd.DataFrame(data=promsy_3[0][0:])

peaksy_rk4, _ = find_peaks(filtery_rk_4, height=360)
promsy_4 = peak_prominences(filtery_rk_4, peaksy_rk4)
promsy_4_df = pd.DataFrame(data=promsy_4[0][0:])

peaksy_rk5, _ = find_peaks(filtery_rk_5, height=330)
promsy_5 = peak_prominences(filtery_rk_5, peaksy_rk5)
promsy_5_df = pd.DataFrame(data=promsy_5[0][0:])

peaksy_rk6, _ = find_peaks(filtery_rk_6, height=325)
promsy_6 = peak_prominences(filtery_rk_6, peaksy_rk6)
promsy_6_df = pd.DataFrame(data=promsy_6[0][0:])

peaksy_rk7, _ = find_peaks(filtery_rk_7, height=330)
promsy_7 = peak_prominences(filtery_rk_7, peaksy_rk7)
promsy_7_df = pd.DataFrame(data=promsy_7[0][0:])

peaksy_rk8, _ = find_peaks(filtery_rk_8, height=300)
promsy_8 = peak_prominences(filtery_rk_8, peaksy_rk8)
promsy_8_df = pd.DataFrame(data=promsy_8[0][0:])

#######  Select Peaks and Calculate Prominences for X - coordinates ###########

max_promsx_1 = promsx_1_df.nlargest(5, 0) #Take 5 max prominences then sort them by index value
max_promsx_2 = promsx_2_df.nlargest(5, 0).sort_index(axis=0)
max_promsx_3 = promsx_3_df.nlargest(10, 0).sort_index(axis=0) #Take 10 max prominences then sort them by index value
max_promsx_4 = promsx_4_df.nlargest(10, 0).sort_index(axis=0)
max_promsx_5 = promsx_5_df.nlargest(10, 0).sort_index(axis=0)
max_promsx_6 = promsx_6_df.nlargest(10, 0).sort_index(axis=0)
max_promsx_7 = promsx_7_df.nlargest(10, 0).sort_index(axis=0)
max_promsx_8 = promsx_8_df.nlargest(10, 0).sort_index(axis=0)


max_promsx_all = pd.concat([max_promsx_1, max_promsx_2,
                            max_promsx_3, max_promsx_4,
                            max_promsx_5, max_promsx_6,
                            max_promsx_7, max_promsx_8], ignore_index = True)


#########  Select Peaks and Calculate Prominences for Y - coordinates #########

max_promsy_1 = promsy_1_df.nlargest(5, 0) #Take 5 max prominences then sort them by index value
max_promsy_2 = promsy_2_df.nlargest(5, 0).sort_index(axis=0)
max_promsy_3 = promsy_3_df.nlargest(10, 0).sort_index(axis=0) #Take 10 max prominences then sort them by index value
max_promsy_4 = promsy_4_df.nlargest(10, 0).sort_index(axis=0)
max_promsy_5 = promsy_5_df.nlargest(10, 0).sort_index(axis=0)
max_promsy_6 = promsy_6_df.nlargest(10, 0).sort_index(axis=0)
max_promsy_7 = promsy_7_df.nlargest(10, 0).sort_index(axis=0)
max_promsy_8 = promsy_8_df.nlargest(10, 0).sort_index(axis=0)


max_promsy_all = pd.concat([max_promsy_1, max_promsy_2,
                            max_promsy_3, max_promsy_4,
                            max_promsy_5, max_promsy_6,
                            max_promsy_7, max_promsy_8], ignore_index = True)


######################## Concatenate Columns ##################################
exp_data = pd.concat([max_promsx_all, max_promsy_all, heights_df, weights_df, 
                      real_measures_df], axis=1)

exp_data_feats = pd.concat([max_promsx_all, max_promsy_all, heights_df, weights_df], axis = 1)

exp_data_lables = real_measures_df

exp_data_feats = exp_data_feats.astype('int')
exp_data_lables = exp_data_lables.astype('int')
           

#Classification Attempt: LDA 
clf = LinearDiscriminantAnalysis()
LinearDiscriminantAnalysis(n_components=4, priors=None, shrinkage=None,
              solver='eigen', store_covariance=False, tol=0.0001)
clf.fit(exp_data_feats, exp_data_lables.values.ravel())
print("LDA Score:", clf.score(exp_data_feats, exp_data_lables.values.ravel()))


#Classification Attempt: Multi-Class Calssification Support Vector Machine
clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
clf.fit(exp_data_feats, exp_data_lables.values.ravel()) 
svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
print("SVM Score:",clf.score(exp_data_feats, exp_data_lables.values.ravel()))


#Classification Attempt: Random Forest
clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
clf.fit(exp_data_feats, exp_data_lables.values.ravel())

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=2, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
print("Random Forest Score:",clf.score(exp_data_feats, exp_data_lables.values.ravel()))


###############################################################################
################# Old but possibly useful code for later #####################
#Idea 4 - Smooth with median filter, then use my method not find peaks 
#bc idk how to get mins: Thanks Ryan McG!


x_vals_df = ankle_df[0]
y_vals_df = ankle_df[1]
filterdx = medfilt(x_vals_df, kernel_size = 5)
peaks, _ = find_peaks(filterdx, height=180)

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



