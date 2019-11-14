# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 21:18:32 2019
@author: palan
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
import operator
import os

#If no path provided the file could be used from current directory path
dirpath = os.getcwd()
print("current directory is : " + dirpath)
#Importing input file
Kin = 5
n=3
#Filename is fixed and provided
filename = "FF62.txt"
datafile= pd.read_csv(filename,header = None, sep='\t')
tot= int(datafile.loc[0,0])
datafile= pd.read_csv(filename,skiprows=[1], sep='\t')
datafile = datafile.reindex(np.random.permutation(datafile.index))
x = datafile.iloc[:,[0,1,2]].values
for i in range(len(x[0])):
	body_values,dorsal_values,type_values = [datafile.iloc[:,[0]].values,datafile.iloc[:,[1]].values,datafile.iloc[:,[2]].values]


#Defining plot method to plot dataset
def plotfish(body_values,dorsal_values,type_values,plotname,xlab,ylab):
    fig = plt.figure()
    #ax = plt.axes()
    #fig.set_size_inches(18,15)
    #for different types plot points with different colors
    for i in range(len(body_values)):
        if(type_values[i] == 1):
            plt.scatter(body_values[i], dorsal_values[i],color='red',marker='D')
        elif(type_values[i] == 0):
            plt.scatter(body_values[i], dorsal_values[i],color='blue',marker='o')
    #labeling x axis and y axis
    plt.title("Fish Plot-"+plotname,weight="bold", size="xx-large")
    plt.xlabel("Body Length "+xlab+"---->",weight="normal", size="large")
    plt.ylabel("Dorsal Length "+ylab+"---->",weight="normal", size="large")
    plt.show()
    #saving the plot as png file
    savename = '/PALCHATTERJEE_ANKITA_'+plotname+'_MyPlot.png'
    fig.savefig(dirpath+savename)

fishplot_before_nor = plotfish(body_values,dorsal_values,type_values,'Raw Data','Raw data(in cm)','Raw data(in cm)')

# Find the min and max values for each column
def columnwise_minmax(x):
	for i in range(len(x[0])):
		col_values = x[:,[i]]
		value_min = float(min(col_values))
		value_max = float(max(col_values))
	return value_min,value_max

# Rescale dataset columns to the range 0-1
def normalize_fishdata(body_values, value_min, value_max):
    for i in range(len(body_values)):
        body_values[i] = (body_values[i] - value_min) / (value_max - value_min)
    return body_values

min_body,max_body = columnwise_minmax(body_values)
min_dorsal,max_dorsal = columnwise_minmax(dorsal_values)

body_values = normalize_fishdata(body_values,min_body,max_body)
dorsal_values = normalize_fishdata(dorsal_values,min_dorsal,max_dorsal)
#Plotting figure after Scaling
fishplot_after_nor = plotfish(body_values,dorsal_values,type_values,'Normalized Data','Normalized data','Normalized data')

# standardize dataset
def standardize_dataset(body_values):
	stdevs = list()
	mean = sum(body_values) / float(len(body_values))
	for i in range(len(body_values)):
        	variance = [pow(body_values[i]-mean, 2)]
        	#print(variance)
	stdevs = [sqrt(sum(variance)/(float(len(body_values)-1)))]
	for j in range(len(body_values)):
		body_values[j] = (body_values[j] - mean)/stdevs
	return body_values

body_values = standardize_dataset(body_values)
dorsal_values = standardize_dataset(dorsal_values)

fishplot_after_std = plotfish(body_values,dorsal_values,type_values,'Standardized Data','Standardized data','Standardized data')


#break the datafile into train and test data
body_train = body_values[0:int(tot*0.8)]
body_test = body_values[int(tot*0.8):tot]
dorsal_train = dorsal_values[0:int(tot*0.8)]
dorsal_test = dorsal_values[int(tot*0.8):tot]
type_train = type_values[0:int(tot*0.8)]
type_test = type_values[int(tot*0.8):tot]

fishplot_train = plotfish(body_train,dorsal_train,type_train,'1 Fold Train Data','1 Fold Train data','1 Fold Train data')
fishplot_test = plotfish(body_test,dorsal_test,type_test,'1 Fold Test Data','1 Fold Test data','1 Fold Test data')

#Break Train data into K Fold
body_train_K = np.split(body_train, 5)
dorsal_train_K = np.split(dorsal_train, 5)
type_train_K = np.split(type_train, 5)


# Fitting Classifier to the Training set
def predict(test_x,test_y,body_values,dorsal_values,type_values):
	distances = list()
	distances_a = list()
	for i in range(len(body_values)):
		distances.append(np.sqrt(pow((body_values[i] - test_x),2) + pow((dorsal_values[i] - test_y),2)))
	new_distances = np.column_stack((distances,type_values))
	distances_a=sorted(new_distances, key=operator.itemgetter(0))
	return distances_a

def predict_type(n,body_values,dorsal_values,type_values,test_x,test_y):
	type1_count = 0
	type0_count = 0
	distance_type = predict(test_x,test_y,body_values,dorsal_values,type_values)
	type_values = [row[1] for row in distance_type]
	type_values_n =type_values[0:n]
	type_values_n = np.array([type_values_n])
	type1_count = np.count_nonzero(type_values_n == 1.0)
	type0_count = n-type1_count
	if(type1_count > type0_count):
		predict_result = '1'
	elif(type1_count < type0_count):
		predict_result = '0'
	return predict_result

def predict_result(n,body_values,dorsal_values,type_values,body_test,dorsal_test):
	predictions = list()
	for i in range(len(body_test)):
		predictions.append(predict_type(n,body_values,dorsal_values,type_values,body_test[i],dorsal_test[i]))
	return np.asarray(predictions)

def confusion_matrix(pred_type_array,type_test):

	print('\n')
	error_count = 0
	#matching_count = 0
	nonmatch_count = 0
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	accuracy = 0
	precision = 0
	accuracy_p = 0
	recall = 0
	F1_score = 0
	#print("Confusion matrix for Type 0 :\n")
	#for i in range(len(type_test)):
	#	if(float(pred_type_array[i]) == float(type_test[i])):
	#		#matching_count = matching_count +1
	#		if((float(pred_type_array[i])==0)):
	#			TP = TP+1
	#		elif((float(pred_type_array[i]==1))):
	#			TN = TN+1
	#	if(float(pred_type_array[i]) != float(type_test[i])):
	#		nonmatch_count = nonmatch_count +1
	#		if((float(pred_type_array[i])==0) and (float(type_test[i])==1)):
	#			FP = FP+1
	#		elif((float(pred_type_array[i])==1) and (float(type_test[i]==0))):
	#			FN = FN+1
	#print('TP:',TP,'TN:',TN,'FP:',FP,'FN:',FN,'\n')
	#error_count = FP+FN
	#accuracy_p = (1-(error_count/int(FP+FN+TP+TN)))*100
	#accuracy = (TP+TN)/(TP+TN+FP+FN)
	#precision = (TP/(TP+FP))
	#recall = (TP/(TP+FN))
	#error_count = 0
	#matching_count = 0
	#nonmatch_count = 0
	#TP = 0
	#TN = 0
	#FP = 0
	#FN = 0
	#error_count = 0
	#accuracy_p = 0
	#precision = 0
	#recall = 0
	print("Confusion matrix for Type 1 :\n")
	for i in type_test:
		print(int(pred_type_array[i]),int(type_test[i]))
		if(int(pred_type_array[i]) == int(type_test[i])):
			#matching_count = matching_count +1
			if((int(pred_type_array[i])==1) and (int(type_test[i])==1)):
				TP = TP+1
			elif((int(pred_type_array[i])==0) and (int(type_test[i])==0)):
				TN = TN+1
		if(int(pred_type_array[i]) != int(type_test[i])):
			nonmatch_count = nonmatch_count +1
			if((int(pred_type_array[i])==1) and (int(type_test[i])==0)):
				FP = FP+1
			elif((int(pred_type_array[i])==0) and (int(type_test[i])==1)):
				FN = FN+1
	print('TP:',TP,'TN:',TN,'FP:',FP,'FN:',FN,'\n')

	error_count = FP+FN
	accuracy_p = (1-(error_count/int(FP+FN+TP+TN)))*100
	accuracy = (TP+TN)/(TP+TN+FP+FN)
	#precision = (TP/(TP+FP))
	#recall = (TP/(TP+FN))
	#F1_score = 2*()
	return error_count,accuracy,accuracy_p,precision,recall
################################################################

for K in range(Kin):
	print('K=',K)
	pred_type_array = predict_result(n,body_train,dorsal_train,type_train,body_train_K[K],dorsal_train_K[K])
	print('\npred_type_array',pred_type_array)
	print('\ntype_train_K',type_train_K[K])

	error_count,accuracy,accuracy_p,precision,recall = confusion_matrix(pred_type_array,type_train_K[K])
	print('error_count:',error_count,'\naccuracy:',accuracy,'\naccuracy_p:',accuracy_p,'\nprecision',precision,recall)

	#confusion_matrix(pred_type_array,type_train_K[4])

