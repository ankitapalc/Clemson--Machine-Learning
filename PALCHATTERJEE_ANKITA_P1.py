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
from mpl_toolkits.mplot3d import Axes3D

#If no path provided the file could be used from current directory path
#n=int(input("Enter k:"))
dirpath = os.getcwd()
print("current directory is : " + dirpath)
#n=int(input('Enter Number of nearest Neighour:'))
#Kin = int(input('Enter Number of Folds::'))
filename = input('Enter File Name::')
#Importing input file
#Filename is fixed and provided
if not filename.strip():
	filename = "FF62.txt"

#############################
#Total count of records
datafile= pd.read_csv(filename,header = None)
tot= int(datafile.loc[0,0])
#Rest of the file
datafile= pd.read_csv(filename,header=None,skiprows=[0], sep='\t')
#Shuffle the file
datafile = datafile.reindex(np.random.permutation(datafile.index))
split_point = int(tot*0.7)
train_data,test_data = datafile[:split_point], datafile[split_point:]
inp_data_train = train_data.iloc[:, [0,1]].values
pred_train = train_data.iloc[:, -1].values
inp_data_test = test_data.iloc[:, [0,1]].values
pred_test = test_data.iloc[:, -1].values
print('inp_data_train',inp_data_train,'pred_train',pred_train,'inp_data_test',inp_data_test,'pred_test',pred_test)

inp_data_train_x1 = inp_data_train[:,0]
inp_data_train_x2 = inp_data_train[:,1]
inp_data_test_x1 = inp_data_test[:,0]
inp_data_test_x2 = inp_data_test[:,1]
##############################

def plotfish(inp_data_train_x1,inp_data_train_x2,pred_train,plotname):
	#fig = plt.figure(figsize = (20,10))
	fig = plt.figure()
	ax=fig.add_subplot(111,projection="3d")
	plt.title(plotname)
	ax.scatter(inp_data_train_x1, inp_data_train_x2, pred_train, c='r',marker='o')#, cmap='hsv')
	ax.set_xlabel('Length of body')
	ax.set_ylabel('Length of Dorsal fin')
	ax.set_zlabel('Fish Type')
	plt.show()
plotfish(inp_data_train_x1,inp_data_train_x2,pred_train,'Raw Data')

# standardize dataset
#######################################################
# Calculate the mean value of a list of numbers
def mean_c(inp_data_train_x1):
	return sum(inp_data_train_x1) / len(inp_data_train_x1)

def standardize(inp_data_train_x1):
	result = np.empty(len(inp_data_train_x1))
	variances = np.linspace(1,len(inp_data_train_x1),len(inp_data_train_x1))
	mean_x1 = mean_c(inp_data_train_x1)
	for i in range(len(inp_data_train_x1)):
		variances[i] = (inp_data_train_x1[i]-mean_x1)**2
	stdev = sqrt(sum(variances)/(len(inp_data_train_x1)))
	for i in range(len(inp_data_train_x1)):
		result[i] = (inp_data_train_x1[i] - mean_x1)/stdev
	return result, mean_x1, stdev

inp_data_train_x1,mean_train_x1,std_train_x1 = standardize(inp_data_train_x1)
inp_data_train_x2,mean_train_x2,std_train_x2 = standardize(inp_data_train_x2)
inp_data_train = inp_data_train_x1,inp_data_train_x2
inp_data_train = np.transpose(np.asarray(inp_data_train))

plotfish(inp_data_train_x1,inp_data_train_x2,pred_train,'Stanadardized data')

#######################################################

# Fitting Classifier to the Training set
def cross_valid(inp_data_train,pred_train):
	cross_valid_point = int(tot*0.7*0.2)
	s_point = 0
	e_point = cross_valid_point
	for i in range(5):
		s_point = i*cross_valid_point#
		e_point = (i+1)*e_point
		inp_data_cv = inp_data_train[s_point:e_point]
		pred_cv = pred_train[s_point:e_point]
		inp_data_tr = inp_data_train[e_point:(tot*0.7)] + inp_data_train[:s_point]
		pred_tr = pred_train[e_point:(tot*0.7)] + inp_data_train[:s_point]
	#return

def predict(inp_data_tr_x1,inp_data_tr_x2,pred_tr,inp_data_cv_x1,inp_data_cv_x2):
	distances = np.empty(shape=[len(inp_data_cv_x1), len(inp_data_tr_x1)])
	dist_type = np.empty(shape=[len(inp_data_cv_x1), len(inp_data_tr_x1)])
#	distances_a = list()
	for i in range(len(inp_data_cv_x1)):
		for j in range(len(inp_data_tr_x1)):
			distances[i][j] = np.sqrt((inp_data_tr_x1[j] - inp_data_cv_x1[i])**2 + (inp_data_tr_x2[j] - inp_data_cv_x2[i])**2)
			dist_type = np.column_stack((distances[i][j],pred_tr[j]))
			#[np.random.randn(3, 4) for _ in range(10)]
	#print('body_values,dorsal_values,distances',test_x,test_y,new_distances)
#	print('before sorting distance',new_distances)
	#distances_a=sorted(new_distances, key=operator.itemgetter(0))
	#print('After sorting distance',distances_a)
	#print('body_values,dorsal_values,distances',test_x,test_y,distances_a)
	return dist_type

#print(predict(inp_data_train_x1,inp_data_train_x2,pred_train,inp_data_test_x1,inp_data_test_x1))

def predict_type(n,inp_data_tr_x1,inp_data_tr_x2,pred_tr,inp_data_cv_x1,inp_data_cv_x2):
	type1_count = 0
	type0_count = 0
	distance_type = predict(inp_data_tr_x1,inp_data_tr_x2,pred_tr,inp_data_cv_x1,inp_data_cv_x2)
	type_values = [row[1] for row in distance_type]
	type_values_n =type_values[0:n]
	#print('type_values_n',type_values_n)
	#type_values_n = np.array([type_values_n])
	type1_count = np.count_nonzero(type_values_n == 1.0)

	type0_count = n-type1_count
	if(type1_count > type0_count):
		predict_result = '1'
	elif(type1_count < type0_count):
		predict_result = '0'
	#print('predict_result',predict_result)
	return predict_result
print(predict_type(n,inp_data_tr_x1,inp_data_tr_x2,pred_tr,inp_data_cv_x1,inp_data_cv_x2))

def predict_result(n,body_values,dorsal_values,type_values,body_test,dorsal_test):
	predictions = list()
	for i in range(len(body_test)):
		predictions.append(predict_type(n,body_values,dorsal_values,type_values,body_test[i],dorsal_test[i]))
	return np.asarray(predictions)

def flat_list_method(l):
	flat_list = []
	for sublist in l:
		for item in sublist:
			flat_list.append(item)
	return flat_list

def confusion_matrix(pred_type_array,type_test):
	error_count = 0
	matching_count = 0
	nonmatch_count = 0
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	#pred_type_array = flat_list_method(pred_type_array)
	#type_test = flat_list_method(type_test)

	print("Confusion matrix for Type 1 :\n")
	for i in type_test:
#		print('\nfloat(pred_type_array[i])',float(pred_type_array[i]),'type_test',float(type_test[i]))
		if(float(pred_type_array[i]) == float(type_test[i])):
			#matching_count = matching_count +1
			if((float(pred_type_array[i])==1) and (float(type_test[i])==1)):
				TP = TP+1
			elif((float(pred_type_array[i])==0) and (float(type_test[i])==0)):
				TN = TN+1
		if(float(pred_type_array[i]) != int(type_test[i])):
			nonmatch_count = nonmatch_count +1
			if((float(pred_type_array[i])==1) and (float(type_test[i])==0)):
				FP = FP+1
			elif((float(pred_type_array[i])==0) and (float(type_test[i])==1)):
				FN = FN+1
	print('TP:',TP,'TN:',TN,'FP:',FP,'FN:',FN,'\n')
	return TP,TN,FP,FN

def count_error_parameters(pred_type_array,type_test):
#count_error_parameters(pred_type_array,type_train_K[K])
	error_count = 0
	accuracy = 0
	precision = 0.000001
	accuracy_p = 0
	recall = 0.000001
	F1_score = 0.000001

	#print('pred_type_array',pred_type_array)
	#print('type_test',type_test)

	TP,TN,FP,FN = confusion_matrix(pred_type_array,type_test)


	error_count = FP+FN
	accuracy_p = (1-(error_count/int(FP+FN+TP+TN)))*100
	accuracy = (TP+TN)/(TP+TN+FP+FN)
	if ((TP+FP)== 0):
		precision = 0.000001
	else:
		precision = (TP/(TP+FP))
	if ((TP+FN)== 0):
		recall = 0.000001
	else:
		recall = (TP/(TP+FN))

	if(precision ==0 and recall == 0 ):
		precision = 0.000001
		recall = 0.000001
	F1_score = 2*((precision*recall)/(precision+recall))
	return error_count,accuracy,accuracy_p,precision,recall,F1_score
########################################################################

#calculate score
def calc_avg_score(Kin,score):
	avg_score = sum(score)/(Kin)
	return avg_score


#def avg_score_per_n(n):
#	#Break Train data into K Fold
#	score = list()
#	body_train_new = list()
#	dorsal_train_new = list()
#	type_train_new = list()
#	body_val_new = list()
#	dorsal_val_new = list()
#	type_val_new = list()
#	body_train_K = np.split(body_train, Kin)
#	dorsal_train_K = np.split(dorsal_train, Kin)
#	type_train_K = np.split(type_train, Kin)
#	pred_type_array = list()

#def best_n():
#	bestn = 0
#	nlist = list(range(1, 11, 2))
#	print('\nneighbors:n',nlist)
#	scores = list()
#	#print('len(nlist):',len(nlist))
#
##	for i in range(len(nlist)):
##		#print('neighbour count to be considered',nlist[i])
##		avg_scr_nlist = avg(nlist[i])
##		#print('avg_score_per_n(nlist[i]):',avg_scr_nlist)
##		scores.append(avg_scr_nlist)
##	print('scores:',scores)
#	j = scores.index(max(scores))
#	bestn = nlist[j]
#	print('max score of scores for all n\'s',max(scores))
#	print('Best n found :',bestn)
#	return bestn


def prompt(n,body_values,dorsal_values,type_values):

	test_x = list()
	test_y = list()
	#pred_type_u = list()
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
			print('Type of Fish:',pred_type)


pred_type_array = predict_result(n,body_train,dorsal_train,type_train,body_test,dorsal_test)

error_count,accuracy,accuracy_p,precision,recall,F1_score = count_error_parameters(pred_type_array,type_train)
print('error_count',error_count,'accuracy',accuracy,'accuracy_p',accuracy_p,precision,recall,F1_score)
prompt(n,body_values,dorsal_values,type_values)


