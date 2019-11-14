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

iterations = 15000
alpha = 0.00001

w0= 0#.0005#0.5,0.0009,0.003,0.00005,0.00000132,0.00001515
w1 = 1#.01
w2 = 0#.01
#w3 = 0.00025
#w4 = 0.3
#w5 = 0.00015

############################################################################
#def normalize(inp_data_train,feature_index):
#	result = np.empty(len(inp_data_train[:,0]))
#	max_value = inp_data_train[:,feature_index].max()
#	#print('max_value',max_value)
#	min_value = inp_data_train[:,feature_index].min()
#	#print('min_value',min_value)
#	result = (inp_data_train[:,feature_index] - min_value) / (max_value - min_value)
#	return result
#inp_data_train_x1 = normalize(inp_data_train,0)
#inp_data_train_x2 = normalize(inp_data_train,1)
#inp_data_train = np.concatenate([(inp_data_train_x1,inp_data_train_x2)])
#print('inp_data_train_normalized',inp_data_train)
#####################################


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
	plt.xlabel("Length of body(in cm)---->",weight="normal", size="large")
	plt.ylabel("Length of dorsal fin(in cm)---->",weight="normal", size="large")
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
##################################################################################

def hypothesis(w0,w1,w2,x1,x2):
	hx = np.empty(len(x1))
	lx = np.empty(len(x1))
	for i in range(len(x1)):
		hx[i] = w0+(w1*x1[i]*x1[i])+(w2*x2[i]*x2[i])
		#print(1/(1+math.e**(-hx[i])))
		lx[i] = 1/(1+pow((math.e),((-1)*(hx[i]))))
		#print('lx',lx[i])
	return lx

###############################################################
def cost(w0,w1,w2,x1,x2,pred_train):
	tot_count = len(pred_train)
	#total_J = 0.0
	Jx = 0.0
	lx = hypothesis(w0,w1,w2,x1,x2)

	for i in range(len(pred_train)):
		if (pred_train[i] == 0):
			#Jx += (-1)*(1/tot_count)*((pred_train[i]*(np.log(lx[i]))+ (1-pred_train[i])*np.log(1 - lx[i])))
			#print('before sum jx',round(Jx,5))
			Jx += np.log(1 - lx[i]+(1e-5))
			#print('jx',round(Jx,5))

		else:
			Jx += np.log(lx[i]+(1e-5))
			#print('jx',Jx)
	Jx = (-1)*(1/tot_count)*Jx
	#print('Jx:',Jx)
	return Jx
#################################################################

def gradient_descent(w0, w1, w2,inp_data_train_x1,inp_data_train_x2, pred_train, alpha, iterations):

	J_history = list()
	iterarr = list()
	w02, w12, w22 = w0, w1, w2
	Jx=0.0
	temp0 = 0.0
	temp1 = 0.0
	temp2 = 0.0
	print('before iteration start w0',w0,'w1',w1,'w2',w2)
	hx = hypothesis(w0,w1,w2,inp_data_train_x1,inp_data_train_x2)
	Jx = cost(w0,w1,w2,inp_data_train_x1,inp_data_train_x2,pred_train)
	print('Initial cost based on hypothesis',Jx)
	J_history.append(Jx)
	iterarr.append(0)
	for j in range(1,iterations):
		J1 = Jx
		for i in range(len(inp_data_train_x1)):
			temp0 += w0 - (1/len(inp_data_train_x1))*alpha*(hx[i]-pred_train[i])
			temp1 += w1 - (1/len(inp_data_train_x1))*alpha*((hx[i]-pred_train[i])*(inp_data_train_x1[i]**2))
			temp2 += w2 - (1/len(inp_data_train_x1))*alpha*((hx[i]-pred_train[i])*(inp_data_train_x2[i]**2))
		#print('temp0',temp0,'temp1',temp1,'temp2',temp2)
		w0 = temp0
		w1 = temp1
		w2 = temp2
		Jx = cost(w0,w1,w2,inp_data_train_x1,inp_data_train_x2,pred_train)
		#print('Cost after alpha reduction (at iteration:',j,') :',Jx)
		J2 = Jx
		#print('\nJ1',J1,'J2',J2)

#		if J1<J2:
#			break
		J_history.append(Jx)
		#print('J_history',J_history)
		iterarr.append(j)

	w02, w12, w22 = w0, w1, w2
	return w02, w12, w22, J_history,iterarr

w0_n,w1_n,w2_n, J_history_n, iterarr_n = gradient_descent(w0, w1, w2,inp_data_train_x1,inp_data_train_x2, pred_train, alpha, iterations)

print('\nafter gradient descent final J value', J_history_n[-1])
plt.scatter(iterarr_n,J_history_n,color='blue',marker='o')
plt.plot(iterarr_n,J_history_n,color='green',marker='D')
#plt.plot(iterarr_n,J_history_n,color='green',marker='D')

plt.xlabel("No of Iterations")
plt.ylabel("J")
plt.show()
######################################################################
def prompt(w0,w1,w2,mean_x1,mean_x2,stdev_x1,stdev_x2):

	inp_data_test_x1 = list()
	inp_data_test_x2 = list()
	while True:
		testval_x = int(input('Enter length of body:'))
		testval_y = int(input('Enter length of dorsal fin:'))
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
			pred_test = hypothesis(w0,w1,w2,inp_data_test_x1,inp_data_test_x2)
			#hypothesis(w0,w1,w2,w3,w4,w5,inp_data_test_x1,inp_data_test_x2)
			#print(w0,w1,w2,w3,w4,w5)
			print('Average score predicted:',round(pred_test,2))
######################################################################

hx = hypothesis(w0_n,w1_n,w2_n,inp_data_train_x1,inp_data_train_x2)
#print('after new hx:',hx,'pred',pred_train,'\nw0',w0_n,'\nw1',w1_n,'\nw2',w2_n,'\nw3',w3_n,'\nw4',w4_n,'\nw5',w5_n)



print('Starting prediction afetr training the model:\n')

inp_data_test_x1_nonstd = inp_data_test_x1
inp_data_test_x2_nonstd = inp_data_test_x2
inp_data_test_x1,mean,std = standardize(inp_data_test_x1)
inp_data_test_x2,mean,std = standardize(inp_data_test_x2)

hx_test = hypothesis(w0_n,w1_n,w2_n,inp_data_test_x1,inp_data_test_x2)
#print('Prediction values from test dataset',pred_test)
#iterations = 10000
#w0_nt,w1_nt,w2_nt,w3_nt,w4_nt,w5_nt, J_history_nt, iterarr_nt =gradient_descent(w0_n,w1_n,w2_n,w3_n,w4_n,w5_n,inp_data_test_x1,inp_data_test_x2, pred_test, alpha, iterations)
cost_test = cost(w0_n,w1_n,w2_n,inp_data_test_x1,inp_data_test_x2,pred_test)
print('\ncost of test data set',cost_test)




plot_fig(inp_data_train_x1,inp_data_train_x2,'Test Data')


print('\nw0',w0_n,'\nw1',w1_n,'\nw2',w2_n)
prompt(w0_n,w1_n,w2_n,mean_train_x1,mean_train_x2,std_train_x1,std_train_x2)