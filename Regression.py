# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 18:49:25 2019
@author: palan
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
import os
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d

# Importing the dataset
dirpath = os.getcwd()
print("current directory is : " + dirpath)
filepath = " "

#If no filepath given as input
filepath = input("Enter file path(If the file is kept on run directory press Enter twice):")
filepath = r'C:\Users\palan\OneDrive\Desktop\Code - Github\machinelearning\Clemson- Machine Learning'
if not filepath.strip():
	filepath = dirpath
#If no Filename given as input, Filename is fixed and provided
filename = input('Enter File Name : ')
if not filename.strip():
	filename = "GPAData.txt"

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
#print('inp_data_train',inp_data_train,'pred_train',pred_train,'inp_data_test',inp_data_test,'pred_test',pred_test)

inp_data_train_x1 = inp_data_train[:,0]
inp_data_train_x2 = inp_data_train[:,1]
inp_data_test_x1 = inp_data_test[:,0]
inp_data_test_x2 = inp_data_test[:,1]

iterations = 10000
alpha = 0.0005

w0= 2#1.5#0.5,0.0009,0.003,0.00005,0.00000132,0.00001515
w1 = 2#1.2
w2 = 0.0005#0.0002
w3 = 0.0005#0.00025
w4 = 0.5#0.28
w5 = 0.0005#0.00012
#print(inp_data_train.shape,pred_train.shape,len(pred_train),inp_data_train_x1)

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
fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax = axes3d(fig)
#ax=fig.add_subplot(2,2,1,projection="3d")
ax = plt.axes(projection='3d')
#ax.plot3D(inp_data_train_x1, inp_data_train_x2, pred_train, 'blue')
a = ax.scatter(inp_data_train_x1, inp_data_train_x2, pred_train, c=pred_train)#, cmap='hsv')
plt.show(a)
#plt.scatter(inp_data_train_x1,pred_train,color='blue',marker='o')
#plt.scatter(inp_data_train_x2,pred_train,color='green',marker='x')
#plt.plot(inp_data_train_x1,pred_train,color='red',marker='o')
#plt.plot(inp_data_train_x2,pred_train,color='red',marker='x')

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
	stdev = sqrt(sum(variances)/(len(inp_data_train_x1)))
	for i in range(len(inp_data_train_x1)):
		result[i] = (inp_data_train_x1[i] - mean_x1)/stdev
	return result, mean_x1, stdev

inp_data_train_x1,mean_train_x1,std_train_x1 = standardize(inp_data_train_x1)
inp_data_train_x2,mean_train_x2,std_train_x2 = standardize(inp_data_train_x2)
#print('mean_train_x1',mean_train_x1,'mean_train_x2',mean_train_x2,'std_train_x1',std_train_x1,'std_train_x2',std_train_x2)

inp_data_train = inp_data_train_x1,inp_data_train_x2
inp_data_train = np.transpose(np.asarray(inp_data_train))
fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.plot3D(inp_data_train_x1, inp_data_train_x2, pred_train, 'blue')
ax.scatter3D(inp_data_train_x1, inp_data_train_x2, pred_train, c=pred_train)#, cmap='hsv')
plt.show()
#plt.scatter(inp_data_train_x1,pred_train,color='blue',marker='o')
#plt.scatter(inp_data_train_x2,pred_train,color='green',marker='x')
#plt.plot(inp_data_train_x1,pred_train,color='green',marker='o')
#plt.plot(inp_data_train_x2,pred_train,color='blue',marker='x')
#plt.show()
#################################################################


##################################################################################
def hypothesis_val(w0,w1,w2,w3,w4,w5,x1,x2):
	#print('inside hypothesis val w0',w0,'w1',w1,'w2',w2,'w3',w3,'w4',w4,'w5',w5)
	hx1 = round((w0+(w1*x1)+(w2*x2)+(w3*x1*x2)+(w4*x1*x1)+(w5*x2*x2)),5)
	#print(hx[i],',',pred_train[i])
	#print('hx:',hx)
	return hx1
def hypothesis(w0,w1,w2,w3,w4,w5,x1,x2):
	hx = np.empty(len(x1))
	print('inside hypothesis w0',w0,'w1',w1,'w2',w2,'w3',w3,'w4',w4,'w5',w5)

	for i in range(len(x1)):
		hx[i] = hypothesis_val(w0,w1,w2,w3,w4,w5,x1[i],x2[i])
		#hx[i] = round((w0+(w1*x1[i])+(w2*x2[i])+(w3*x1[i]*x2[i])+(w4*x1[i]*x1[i])+(w5*x2[i]*x2[i])),5)
		#print(hx[i],',',pred_train[i])
	#print('hx:',hx)
	return hx
#################################################################
def cost(w0,w1,w2,w3,w4,w5,x1,x2,pred_train):
	tot_count = len(pred_train)
	total_J = 0.0
	Jx = 0.0
	hx = hypothesis(w0,w1,w2,w3,w4,w5,x1,x2)
	#print('inside cost w0',w0,'w1',w1,'w2',w2,'w3',w3,'w4',w4,'w5',w5)

	for i in range(len(pred_train)):
		total_J = total_J+ ((hx[i]-pred_train[i])*(hx[i]-pred_train[i]))
	Jx = round((1/(2*tot_count))*total_J,5)
	#print('Jx:',Jx)
	return Jx
#################################################################

def gradient_descent(w0, w1, w2, w3, w4, w5,inp_data_train_x1,inp_data_train_x2, pred_train, alpha, iterations):

	J_history = list()
	#print('Empty J',J_history)
	iterarr = list()
	w02, w12, w22, w32, w42, w52 = w0, w1, w2, w3, w4, w5
	Jx=0.0
	temp0 = 0.0
	temp1 = 0.0
	temp2 = 0.0
	temp3 = 0.0
	temp4 = 0.0
	temp5 = 0.0
	#print('before iteration start w0',w0,'w1',w1,'w2',w2,'w3',w3,'w4',w4,'w5',w5)
	hx = hypothesis(w0,w1,w2,w3,w4,w5,inp_data_train_x1,inp_data_train_x2)
	Jx = round(cost(w0,w1,w2,w3,w4,w5,inp_data_train_x1,inp_data_train_x2,pred_train),5)
	print('Initial cost based on hypothesis',Jx)
	J_history.append(Jx)
	iterarr.append(0)
	for j in range(1,iterations):
		#print('iteration start w0',w0,'w1',w1,'w2',w2,'w3',w3,'w4',w4,'w5',w5)
		J1 = Jx
		for i in range(len(inp_data_train_x1)):
			temp0 = temp0 + (hx[i]-pred_train[i])
			temp1 = temp1 + ((hx[i]-pred_train[i])*inp_data_train_x1[i])
			temp2 = temp2 + ((hx[i]-pred_train[i])*inp_data_train_x2[i])
			temp3 = temp3 + ((hx[i]-pred_train[i])*inp_data_train_x1[i]*inp_data_train_x2[i])
			temp4 = temp4 + ((hx[i]-pred_train[i])*inp_data_train_x1[i]*inp_data_train_x1[i])
			temp5 = temp5 + ((hx[i]-pred_train[i])*inp_data_train_x2[i]*inp_data_train_x2[i])
		#print('temp0',(alpha/(len(inp_data_train_x1)))*temp0,'temp1',(alpha/(len(inp_data_train_x1)))*temp1,'temp2',(alpha/(len(inp_data_train_x1)))*temp2,'temp3',(alpha/(len(inp_data_train_x1)))*temp3,'temp4',(alpha/(len(inp_data_train_x1)))*temp4,'temp5',(alpha/(len(inp_data_train_x1)))*temp5)
		w0 = w0 - (alpha/(len(inp_data_train_x1)))*temp0
		w1 = w1 - (alpha/(len(inp_data_train_x1)))*temp1
		w2 = w2 - (alpha/(len(inp_data_train_x1)))*temp2
		w3 = w3 - (alpha/(len(inp_data_train_x1)))*temp3
		w4 = w4 - (alpha/(len(inp_data_train_x1)))*temp4
		w5 = w5 - (alpha/(len(inp_data_train_x1)))*temp5
		#print('\nw0',w0,'\nw1',w1,'\nw2',w2,'\nw3',w3,'\nw4',w4,'\nw5',w5)
		Jx = cost(w0,w1,w2,w3,w4,w5,inp_data_train_x1,inp_data_train_x2,pred_train)
		print('Cost after alpha reduction (at iteration:',j,') :',Jx)
		J2 = Jx
		print('\nJ1',J1,'J2',J2)

		if J1<J2:
			break
		J_history.append(Jx)
		iterarr.append(j)

		w02, w12, w22, w32, w42, w52 = w0, w1, w2, w3, w4, w5
	return w02, w12, w22, w32, w42, w52, J_history,iterarr

w0_n,w1_n,w2_n,w3_n,w4_n,w5_n, J_history_n, iterarr_n = gradient_descent(w0, w1, w2, w3, w4, w5,inp_data_train_x1,inp_data_train_x2, pred_train, alpha, iterations)

print('\nafter gradient descent w0',w0_n,'\nw1',w1_n,'\nw2',w2_n,'\nw3',w3_n,'\nw4',w4_n,'\nw5',w5_n,'\nJx_history', J_history_n)
plt.scatter(iterarr_n,J_history_n,color='blue',marker='o')
plt.plot(iterarr_n,J_history_n,color='green',marker='D')
plt.show()
######################################################################
def prompt(w0,w1,w2,w3,w4,w5,mean_x1,mean_x2,stdev_x1,stdev_x2):

	inp_data_test_x1 = list()
	inp_data_test_x2 = list()
	while True:
		testval_x = int(input('Enter hours of study:'))
		testval_y = int(input('Enter no of beers consumed:'))
		if(testval_x==0 and testval_y == 0):
			break
		else:
#			inp_data_test_x1.append(testval_x)
#			inp_data_test_x2.append(testval_y)
			inp_data_test_x1 = testval_x
			inp_data_test_x2 = testval_y
			#########################################################
#
			print('mean_x1',mean_x1,'mean_x2',mean_x1,'stdev_x1',stdev_x1,'stdev_x2',stdev_x2)
			inp_data_test_x1 = (inp_data_test_x1 - mean_x1)/stdev_x1
			inp_data_test_x2 = (inp_data_test_x2 - mean_x2)/stdev_x2
			#########################################################
			pred_test = hypothesis_val(w0,w1,w2,w3,w4,w5,inp_data_test_x1,inp_data_test_x2)
			#hypothesis(w0,w1,w2,w3,w4,w5,inp_data_test_x1,inp_data_test_x2)
			print(w0,w1,w2,w3,w4,w5)
			print('Score predicted:',pred_test)
######################################################################

hx = hypothesis(w0_n,w1_n,w2_n,w3_n,w4_n,w5_n,inp_data_train_x1,inp_data_train_x2)
print('after new hx:',hx,'pred',pred_train,'\nw0',w0_n,'\nw1',w1_n,'\nw2',w2_n,'\nw3',w3_n,'\nw4',w4_n,'\nw5',w5_n)



print('Starting prediction afetr training the model:\n')

inp_data_test_x1,mean,std = standardize(inp_data_test_x1)
inp_data_test_x2,mean,std = standardize(inp_data_test_x2)

#pred_test = np.empty(len(inp_data_test_x1))

pred_test = hypothesis(w0_n,w1_n,w2_n,w3_n,w4_n,w5_n,inp_data_test_x1,inp_data_test_x2)
print('Prediction values from test dataset',pred_test)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(inp_data_test_x1, inp_data_test_x2, pred_test, c=pred_test)#, cmap='hsv')
ax.plot3D(inp_data_test_x1, inp_data_test_x2, pred_test, 'blue')
plt.show()

print('\nbefore prompt w0',w0_n,'\nw1',w1_n,'\nw2',w2_n,'\nw3',w3_n,'\nw4',w4_n,'\nw5',w5_n,'std_train_x1',std_train_x1)
prompt(w0_n,w1_n,w2_n,w3_n,w4_n,w5_n,mean_train_x1,mean_train_x2,std_train_x1,std_train_x2)