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
#n=int(input('Enter Number of nearest Neighour:'))
Kin = int(input('Enter Number of Folds::'))
filename = input('Enter File Name::')
#Importing input file
#Filename is fixed and provided
if not filename.strip():
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
#plotfish(test_x,test_y,pred_type_u,'User Input data','User Input data','User Input data')
    fig = plt.figure()
    ax = plt.axes()
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
	stdevs = [float(sqrt(sum(variance)/(len(body_values)-1)))]
	for j in range(len(body_values)):
		body_values[j] = (float(body_values[j]) - mean/stdevs)
	return body_values

body_values = standardize_dataset(body_values)
dorsal_values = standardize_dataset(dorsal_values)
fishplot_after_std = plotfish(body_values,dorsal_values,type_values,'Standardized Data','Standardized data','Standardized data')

def flat_list_method(l):
	flat_list = []
	for sublist in l:
		for item in sublist:
			flat_list.append(item)
	return flat_list
#break the datafile into train and test data
body_train = body_values[0:int(tot*0.8)]
body_test = body_values[int(tot*0.8):tot]
dorsal_train = dorsal_values[0:int(tot*0.8)]
dorsal_test = dorsal_values[int(tot*0.8):tot]
type_train = type_values[0:int(tot*0.8)]
type_test = type_values[int(tot*0.8):tot]

fishplot_train = plotfish(body_train,dorsal_train,type_train,'1 Fold Train Data','1 Fold Train data','1 Fold Train data')
fishplot_test = plotfish(body_test,dorsal_test,type_test,'1 Fold Test Data','1 Fold Test data','1 Fold Test data')



def predict(body_values,dorsal_values,type_values,test_x,test_y):
##body_train_K[K][i],dorsal_train_K[K][i],body_train_new,dorsal_train_new,type_train_new)
	distances = list()
	distances_a = list()
#	type_values = np.array(type_values)
#	type_values = flat_list_method(type_values)
#	test_x = flat_list_method(test_x)
#	test_x = flat_list_method(test_x)
#	test_y = flat_list_method(test_y)
#	test_y = flat_list_method(test_y)
	#print('test_x',test_x)
#	print('body_values inside predict',body_values)
#	#print('test_y',test_y)
#	print('type_values',type_values)
#	#type_values = flat_list_method(type_values)
#	#type_values = flat_list_method(type_values)
	for i in range(len(body_values)):
		#print(len(body_values))
		#for j in range(len(test_x)):
		distances.append(np.sqrt(pow((body_values[i] - test_x),2)))#+ pow((dorsal_values[i] - test_y),2) ))

	#print('distances',len(distances))
	distances = np.asarray(distances)

	new_distances = np.column_stack((distances,type_values))
	#distances = flat_list_method(distances)
	#distances = flat_list_method(distances)
	#type_values = flat_list_method(type_values)
#	print('len(distances)',len(distances),distances,'type_values',type_values)
#	print(type(distances),type(type_values))
	distances = np.asarray(distances)


	#print('new_distances',new_distances)
	distances_a=sorted(new_distances, key=operator.itemgetter(0))
	return distances_a#shud return 16
#predict(body_train_K[K][i],dorsal_train_K[K][i],body_train,dorsal_train,type_train,body_train_K[K][i],dorsal_train_K[K][i])


def predict_type(n,body_values,dorsal_values,type_values,test_x,test_y):
	#n,body_train_new,dorsal_train_new,type_train_new,body_train_K[K],dorsal_train_K[K])
	#print('test_x',test_x)
	distance_type = list()
	type1_count=0
	#for i in range(len(test_x)):
#	print('len(test_x)',len(test_x))
#	distance_type = predict(body_values,dorsal_values,type_values,test_x[i],test_y[i])
	distance_type = predict(body_values,dorsal_values,type_values,test_x,test_y)#shud return 16 records
	#print('distance_type',distance_type)
	type_values = [row[1] for row in distance_type]
		#print('type_values:',type_values)
	type_values_n =type_values[0:n]
	#print('type_values_n',type_values_n)
	#print('type_values_n',type_values_n)

	type1_count = np.count_nonzero(type_values_n)
	#print('type1_count',type1_count)
	type0_count = n-type1_count
	if(type1_count > type0_count):
		predict_result = '1'
	elif(type1_count < type0_count):
		predict_result = '0'
	return predict_result#shall give 1 prediction
#predict_type(n,body_train,dorsal_train,type_train,body_train_K[K][i],dorsal_train_K[K][i])
def predict_result(n,body_values,dorsal_values,type_values,body_test,dorsal_test):
#predict_result(n,body_train_new,dorsal_train_new,type_train_new,body_train_K[K],dorsal_train_K[K])
	#print('body_val_new',body_test)

	predictions = list()
	type_values = flat_list_method(type_values)
	#print('type_values in pred_res:',type_values)
	#print('len(body_values)',body_test)

	for i in range(len(dorsal_test)):
		#print('body values:',body_values,'dorsal_values',dorsal_values,'type_values',type_values,'body_test',body_test,'dorsal_test',dorsal_test)
		predictions.append(predict_type(n,body_values,dorsal_values,type_values,body_test[i],dorsal_test[i]))
	print('predict result predictions:',predictions)
	return np.asarray(predictions)#shall give 16 records

def confusion_matrix(pred_type_array,type_test):

	nonmatch_count = 0
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	pred_type_array = flat_list_method(pred_type_array)
#	type_test = flat_list_method(type_test)
	print('pred_type_array',pred_type_array)
	print('type_test',type_test)
	print("Confusion matrix for Type 1 :\n")
	for i in range(len(type_test)):
		#print('pred_type_array',int(pred_type_array[i]),'type_test',int(type_test[i]))
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
	return TP,TN,FP,FN



def count_error_parameters(pred_type_array,type_test):
#count_error_parameters(pred_type_array,type_train_K[K])
	error_count = 0
	accuracy = 0
	precision = 0
	accuracy_p = 0
	recall = 0
	F1_score = 0

	#print('pred_type_array',pred_type_array)
	#print('type_test',type_test)

	TP,TN,FP,FN = confusion_matrix(pred_type_array,type_test)

	error_count = FP+FN
	accuracy_p = (1-(error_count/int(FP+FN+TP+TN)))*100
	accuracy = (TP+TN)/(TP+TN+FP+FN)
	precision = (TP/(TP+FP))
	recall = (TP/(TP+FN))
	F1_score = 2*((precision*recall)/(precision+recall))
	return error_count,accuracy,accuracy_p,precision,recall,F1_score
########################################################################

#calculate score
def calc_avg_score(Kin,score):
	avg_score = sum(score)/(Kin)
	return avg_score


def avg_score_per_n(n):
	#Break Train data into K Fold
	score = list()
	body_train_new = list()
	dorsal_train_new = list()
	type_train_new = list()
	body_val_new = list()
	dorsal_val_new = list()
	type_val_new = list()
	body_train_K = np.split(body_train, Kin)
	dorsal_train_K = np.split(dorsal_train, Kin)
	type_train_K = np.split(type_train, Kin)
	pred_type_array = list()


	body_train_K1 = np.asarray(body_train_K)
	dorsal_train_K1 = np.asarray(dorsal_train_K)
	type_train_K1 = np.asarray(type_train_K)

	for K in range(Kin):

		print('Entering Fold=',K,'neighbour',n)
		if(K==0):
			body_train_new = np.concatenate([body_train_K1[1:2],body_train_K1[2:5]])
			body_val_new = body_train_K1[0:1]
			dorsal_train_new = np.concatenate([dorsal_train_K1[1:2],dorsal_train_K1[2:5]])
			dorsal_val_new = dorsal_train_K1[0:1]
			type_train_new = np.concatenate([type_train_K1[1:2],type_train_K1[2:5]])
			type_val_new = type_train_K1[0:1]
		elif(K==1):
			body_train_new = np.concatenate([body_train_K1[0:1],body_train_K1[2:5]])
			body_val_new = body_train_K1[1:2]
			dorsal_train_new = np.concatenate([dorsal_train_K1[0:1],dorsal_train_K1[2:5]])
			dorsal_val_new = dorsal_train_K1[1:2]
			type_train_new = np.concatenate([type_train_K1[0:1],type_train_K1[2:5]])
			type_val_new = type_train_K1[1:2]
		elif(K==2):
			body_train_new = np.concatenate([body_train_K1[0:2],body_train_K1[3:5]])
			body_val_new = body_train_K1[2:3]
			dorsal_train_new = np.concatenate([dorsal_train_K1[0:2],dorsal_train_K1[3:5]])
			dorsal_val_new = dorsal_train_K1[2:3]
			type_train_new = np.concatenate([type_train_K1[0:2],type_train_K1[3:5]])
			type_val_new = type_train_K1[2:3]
		elif(K==3):
			body_train_new = np.concatenate([body_train_K1[0:3],body_train_K1[4:5]])
			body_val_new = body_train_K1[3:4]
			dorsal_train_new = np.concatenate([dorsal_train_K1[0:3],dorsal_train_K1[4:5]])
			dorsal_val_new = dorsal_train_K1[3:4]
			type_train_new = np.concatenate([type_train_K1[0:3],type_train_K1[4:5]])
			type_val_new = type_train_K1[3:4]
		elif(K==4):
			body_train_new = np.concatenate([body_train_K1[0:2],body_train_K1[2:4]])
			body_val_new = body_train_K1[4:5]
			dorsal_train_new = np.concatenate([dorsal_train_K1[0:2],dorsal_train_K1[2:4]])
			dorsal_val_new = dorsal_train_K1[4:5]
			type_train_new = np.concatenate([type_train_K1[0:2],type_train_K1[2:4]])
			type_val_new = type_train_K1[4:5]
		#print('body_val_new',body_val_new)

		body_val_new=flat_list_method(body_val_new)
		body_val_new=flat_list_method(body_val_new)



		body_train_new=flat_list_method(body_train_new)
		dorsal_train_new=flat_list_method(dorsal_train_new)
		#type_train_new=flat_list_method(type_train_new)
		body_train_new=flat_list_method(body_train_new)
		dorsal_train_new=flat_list_method(dorsal_train_new)
		#type_train_new=flat_list_method(type_train_new)
		#16 rec
		#print('body_val_new elements:',body_val_new[0])
		pred_type_array = predict_result(n,body_train_new,dorsal_train_new,type_train_new,body_val_new,dorsal_val_new)
		#print('pred_type_array_after_result',pred_type_array)
		error_count,accuracy,accuracy_p,precision,recall,F1_score = count_error_parameters(pred_type_array,type_val_new[0][K])
		#print_confu_matrix(pred_type_array,type_val_new)
		#print('error_count per K:',error_count,'\naccuracy per K:',accuracy,'\naccuracy_p per K:',accuracy_p,'\nprecision per K',precision,'\nrecall per K',recall,'\nF1_score per K',F1_score)
		#score.append(F1_score)
	#calc avg of all K scores for each n
	print('score:',score)
	avg_score_n = calc_avg_score(Kin,score)
	print('avg_score for n=',n,'is :',avg_score_n)
	#print_confu_matrix(pred_type_array,type_test)
	return avg_score_n

def best_n():
	bestn = 0
	nlist = list(range(1, 11, 2))
	print('\nneighbors:n',nlist)
	scores = list()
	#print('len(nlist):',len(nlist))

	for i in range(len(nlist)):
		#print('neighbour count to be considered',nlist[i])
		avg_scr_nlist = avg_score_per_n(nlist[i])
		#print('avg_score_per_n(nlist[i]):',avg_scr_nlist)
		scores.append(avg_scr_nlist)
	print('scores:',scores)
	j = scores.index(max(scores))
	bestn = nlist[j]
	print('max score of scores for all n\'s',max(scores))
	print('Best n found :',bestn)
	return bestn


def prompt(n,body_values,dorsal_values,type_values):

	test_x = list()
	test_y = list()
	pred_type_u = list()
	print('Number of neighbours considered is :',n)
	while True:
		testval_x = float(input('Enter body length in cm:'))
		testval_y = float(input('Enter dorsal length in cm:'))
		if(testval_x==0 and testval_y == 0):
			break
		else:
			test_x.append(testval_x)
			test_y.append(testval_y)
			pred_type = predict_type(n,body_values,dorsal_values,type_values,testval_x,testval_y)
			pred_type_u.append(pred_type)
			print('Type of Fish:',pred_type)


prompt(best_n(),body_values,dorsal_values,type_values)

