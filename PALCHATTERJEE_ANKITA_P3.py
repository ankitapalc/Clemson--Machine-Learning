# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 18:49:25 2019
@author: palan
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
#from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import random


# Importing the dataset
dirpath = os.getcwd()
print("current directory is : " + dirpath)
filepath = " "

#If no filepath given as input
filepath = input("Enter file path(If the file is kept on run directory press Enter twice):")
#filepath = r'C:\Users\palan\OneDrive\Desktop\Code - Github\machinelearning\Clemson- Machine Learning'
if not filepath.strip():
	filepath = dirpath
#If no Filename given as input, Filename is fixed and provided
filename = input('Enter File Name : ')
if not filename.strip():
	filename = "FF62.txt"

filename = filepath+"\\"+filename
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

inp_data_train_x1 = inp_data_train[:,0]
inp_data_train_x2 = inp_data_train[:,1]
inp_data_test_x1 = inp_data_test[:,0]
inp_data_test_x2 = inp_data_test[:,1]

iterations = 5000
alpha = 1e-3

w0= 0.5#-1#2#.0005#0.5,0.0009,0.003,0.00005,0.00000132,0.00001515
w1 = 0.4#1#.01
w2 = -0.2#-1#.01
w3 = 0.1
w4 = -0.3#.03
#w5 = 0.6#.0015

def plot_fig(inp_data_train_x1,inp_data_train_x2,title):
	fig = plt.figure()
	ax = plt.axes()
	#fig.set_size_inches(20,15)

	for i in range(len(inp_data_train_x1)):
		if(pred_train[i] == 0):
			red = plt.scatter(inp_data_train_x1[i], inp_data_train_x2[i],color='red',marker='D')
		elif(pred_train[i] == 1):
			green = plt.scatter(inp_data_train_x1[i], inp_data_train_x2[i],color='green',marker='x')
	#labeling x axis and y axis
	plt.title(title,weight="bold", size="xx-large")
	plt.xlabel("Length of body---->",weight="normal", size="large")
	plt.ylabel("Length of dorsal fin---->",weight="normal", size="large")
	plt.legend([red, green], ["Fish Type 0", "Fish Type 1"])
	plt.show()

plot_fig(inp_data_train_x1,inp_data_train_x2,'Raw Data')
#################################################################
# Calculate the mean value of a list of numbers
def mean_c(inp_data_train_x1):
	return sum(inp_data_train_x1) / len(inp_data_train_x1)

def standardize(inp_data_train_x1):
	result = np.empty(len(inp_data_train_x1))
	variances = np.linspace(1,len(inp_data_train_x1),len(inp_data_train_x1))
	mean_x1 = mean_c(inp_data_train_x1)
	for i in range(len(inp_data_train_x1)):
		variances[i] = (inp_data_train_x1[i]-mean_x1)**2
	stdev = np.sqrt(sum(variances)/(len(inp_data_train_x1)))
	for i in range(len(inp_data_train_x1)):
		result[i] = (inp_data_train_x1[i] - mean_x1)/stdev
	return result, mean_x1, stdev

inp_data_train_x1,mean_train_x1,std_train_x1 = standardize(inp_data_train_x1)
inp_data_train_x2,mean_train_x2,std_train_x2 = standardize(inp_data_train_x2)
inp_data_train = inp_data_train_x1,inp_data_train_x2
inp_data_train = np.transpose(np.asarray(inp_data_train))
##################################################################
plot_fig(inp_data_train_x1,inp_data_train_x2,'Standardized Data')
#print(inp_data_train,pred_train)
##################################################################################
def hypothesis(w0,w1,w2,w3,w4,x1,x2):
	hx = np.empty(len(x1))
	lx = np.empty(len(x1))
	for i in range(len(x1)):
		hx[i] = w0+(w3*x1[i])+(w4*x2[i])+(w1*(x1[i]**2))+(w2*(x2[i]**2))#+(w3*x1[i]*x2[i])
		lx[i] = 1/(1+(np.exp((-1)*(hx[i]))))
	return lx

###############################################################
def cost(w0,w1,w2,w3,w4,x1,x2,pred_train,lx):
	tot_count = len(pred_train)
	Jx = 0.0
	#lx = hypothesis(w0,w1,w2,w3,w4,w5,x1,x2)
	small_val = (1e-5)
	for i in range(len(pred_train)):
		Jx +=  (-1)*(pred_train[i]*(np.log(lx[i])+small_val) + (1-pred_train[i])*(np.log(1-lx[i]+small_val)))
	Jx1 = (1/tot_count)*Jx
	return Jx1
#################################################################

def gradient_descent(w0, w1, w2,w3,w4,inp_data_train_x1,inp_data_train_x2, pred_train, alpha, iterations):
	J_history = list()
	iterarr = list()
	w02, w12, w22, w32,w42 = w0, w1, w2, w3,w4
#	tot_length = len(pred_train)
	Jx=0.0
	print('before iteration start w0',w0,'w1',w1,'w2',w2, 'w3',w3,'w4',w4)
	hx = hypothesis(w0,w1,w2,w3,w4,inp_data_train_x1,inp_data_train_x2)
	Jx = cost(w0,w1,w2,w3,w4,inp_data_train_x1,inp_data_train_x2,pred_train,hx)
	print('Initial cost based on hypothesis',Jx)
	J_history.append(Jx)
	iterarr.append(0)
	for j in range(1,iterations):
		J1 = Jx
		for i in range(len(inp_data_train_x1)):
			w0 = w0 - np.sum(alpha*(hx[i]-pred_train[i]))
			w1 = w1 - np.sum(alpha*((hx[i]-pred_train[i])*(inp_data_train_x1[i]**2)))
			w2 = w2 - np.sum(alpha*((hx[i]-pred_train[i])*(inp_data_train_x2[i]**2)))
			w3 = w3 - np.sum(alpha*((hx[i]-pred_train[i])*(inp_data_train_x1[i])))
			w4 = w4 - np.sum(alpha*((hx[i]-pred_train[i])*(inp_data_train_x2[i])))

		hx = hypothesis(w0,w1,w2,w3,w4,inp_data_train_x1,inp_data_train_x2)
		Jx = cost(w0,w1,w2,w3,w4,inp_data_train_x1,inp_data_train_x2,pred_train,hx)
		J2 = Jx

		if J1<J2:
			break
		J_history.append(Jx)
		iterarr.append(j)
	w02, w12, w22, w32,w42 = w0, w1, w2, w3,w4
	return w02, w12, w22,w32,w42,J_history,iterarr

w0_n,w1_n,w2_n,w3_n,w4_n,J_history_n, iterarr_n = gradient_descent(w0, w1, w2,w3,w4,inp_data_train_x1,inp_data_train_x2, pred_train, alpha, iterations)

print('\nafter gradient descent final J value', J_history_n[-1])
plt.scatter(iterarr_n,J_history_n,color='blue',marker='o')
plt.plot(iterarr_n,J_history_n,color='green',marker='o')
plt.xlabel("No of Iterations")
plt.ylabel("Cost")
plt.show()

def predict_type(w0_n,w1_n,w2_n,w3_n,w4_n,inp_data_train_x1,inp_data_train_x2):
	class_type = np.empty(len(inp_data_train_x1))
	hx_prmt = hypothesis(w0_n,w1_n,w2_n,w3_n,w4_n,inp_data_train_x1,inp_data_train_x2)
	for i in range(len(hx_prmt)):
		if (hx_prmt[i]>=0.5):
			class_type[i] = 1
		else:
			class_type[i] = 0
	return class_type

######################################################################
def confusion_matrix(pred_train,inp_data_train_x1,inp_data_train_x2,w0_n,w1_n,w2_n,w3_n,w4_n):
	class_type = predict_type(w0_n,w1_n,w2_n,w3_n,w4_n,inp_data_train_x1,inp_data_train_x2)
	nonmatch_count = 0
	TP = 0
	TN = 0
	FP = 0
	FN = 0

	print("Confusion matrix for Type 1 :\n")
	for i in range(len(pred_train)):
		if(float(pred_train[i]) == float(class_type[i])):
			if((float(pred_train[i])==1) and (float(class_type[i])==1)):
				TP = TP+1
			elif((float(pred_train[i])==0) and (float(class_type[i])==0)):
				TN = TN+1
		if(float(pred_train[i]) != float(class_type[i])):
			nonmatch_count = nonmatch_count +1
			if((float(pred_train[i])==1) and (float(class_type[i])==0)):
				FP = FP+1
			elif((float(pred_train[i])==0) and (float(class_type[i])==1)):
				FN = FN+1
	print('TP:',TP,'TN:',TN,'FP:',FP,'FN:',FN,'\n')
	return TP,TN,FP,FN

def count_error_parameters(pred_train,inp_data_train_x1,inp_data_train_x2,w0_n,w1_n,w2_n,w3_n,w4_n):
	error_count = 0
	accuracy = 0
	precision = 0#.000001
	accuracy_p = 0
	recall = 0#.000001
	F1_score = 0#.000001

	TP,TN,FP,FN = confusion_matrix(pred_train,inp_data_train_x1,inp_data_train_x2,w0_n,w1_n,w2_n,w3_n,w4_n)

	error_count = FP+FN
	accuracy_p = (1-(error_count/(FP+FN+TP+TN)))*100
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
	print('\nerror_count',error_count,'\naccuracy',accuracy,'\naccuracy_p',accuracy_p,'\nprecision',precision,'\nrecall',recall,'\nF1_score',F1_score)
	return error_count,accuracy,accuracy_p,precision,recall,F1_score

######################################################################
def prompt(w0,w1,w2,w3_n,w4_n,mean_x1,mean_x2,stdev_x1,stdev_x2):

#	inp_data_test_x1 = list()
#	inp_data_test_x2 = list()
	while True:
		testval_x = float(input('Enter length of body(in cm):'))
		testval_y = float(input('Enter length of dorsal fin(in cm):'))
		if(testval_x==0 and testval_y == 0):
			break
		else:
#			inp_data_test_x1.append(testval_x)
#			inp_data_test_x2.append(testval_y)
			inp_data_test_x1 = testval_x
			inp_data_test_x2 = testval_y
			#########################################################
#
			#print('mean_x1',mean_x1,'mean_x2',mean_x1,'stdev_x1',stdev_x1,'stdev_x2',stdev_x2)
			inp_data_test_x1 = (inp_data_test_x1 - mean_x1)/stdev_x1
			inp_data_test_x2 = (inp_data_test_x2 - mean_x2)/stdev_x2
			#########################################################
			inp_data_test_x1 = np.asarray(inp_data_test_x1)* np.ones(1)
			inp_data_test_x2 = np.asarray(inp_data_test_x1)* np.ones(1)
			#pred_test = hypothesis(w0,w1,w2,w3_n,w4_n,w5_n,inp_data_test_x1,inp_data_test_x2)
			pred_prmt = predict_type(w0_n,w1_n,w2_n,w3_n,w4_n,inp_data_test_x1,inp_data_test_x2)
			print('Type predicted:',int(pred_prmt))
######################################################################

#hx = hypothesis(w0_n,w1_n,w2_n,w3_n,w4_n,w5_n,inp_data_train_x1,inp_data_train_x2)

print('Starting prediction afetr training the model:\n')

inp_data_test_x1_nonstd = inp_data_test_x1
inp_data_test_x2_nonstd = inp_data_test_x2
inp_data_test_x1,mean,std = standardize(inp_data_test_x1_nonstd)
inp_data_test_x2,mean,std = standardize(inp_data_test_x2_nonstd)

hx_test = hypothesis(w0_n,w1_n,w2_n,w3_n,w4_n,inp_data_test_x1,inp_data_test_x2)
#print('hx_test',hx_test)
cost_test = cost(w0_n,w1_n,w2_n,w3_n,w4_n,inp_data_test_x1,inp_data_test_x2,pred_test,hx_test)
print('\ncost of test data set',cost_test)
plot_fig(inp_data_test_x1,inp_data_test_x2,'Test Data')
count_error_parameters(pred_test,inp_data_test_x1,inp_data_test_x2,w0_n,w1_n,w2_n,w3_n,w4_n)
print('\nw0',w0_n,'\nw1',w1_n,'\nw2',w2_n,'\nw3',w3_n,'\nw4',w4_n)
prompt(w0,w1,w2,w3_n,w4_n,mean_train_x1,mean_train_x2,std_train_x1,std_train_x2)