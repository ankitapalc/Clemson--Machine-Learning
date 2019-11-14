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

datafile=open('FF62.txt')
datafile= pd.read_csv('FF62.txt',header = None, sep='\t')
tot= int(datafile.loc[0,0])
datafile= pd.read_csv('FF62.txt',skiprows=[1], sep='\t')
x = datafile.iloc[:,[0,1,2]].values
for i in range(len(x[0])):
	body_values,dorsal_values,type_values = [datafile.iloc[:,[0]].values,datafile.iloc[:,[1]].values,datafile.iloc[:,[2]].values]




body_train = body_values[0:int(tot*0.8)]
body_test = body_values[int(tot*0.8):tot]
dorsal_train = dorsal_values[0:int(tot*0.8)]
dorsal_test = dorsal_values[int(tot*0.8):tot]
type_train = type_values[0:int(tot*0.8)]
type_test = type_values[int(tot*0.8):tot]

print(body_test,dorsal_test,type_test)

##########################################################

neighbors = list(range(1, 50, 2))
print('\nneighbors',neighbors)
# empty list that will hold cv scores
cv_scores = []

# perform 5-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

    # changing to misclassification error
mse = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[mse.index(min(mse))]
print("The optimal number of neighbors is {}".format(optimal_k))

# plot misclassification error vs k
plt.plot(neighbors, mse)
plt.xlabel("Number of Neighbors K")
plt.ylabel("Misclassification Error")
plt.show()

##########################################################

def predict(test_x,test_y,body_values,dorsal_values,type_values):
	distances = list()
	distances_a = list()
	for i in range(len(body_values)):
		distances.append(np.sqrt(pow((body_values[i] - test_x),2) + pow((dorsal_values[i] - test_y),2)))
	new_distances = np.column_stack((distances,type_values))
	#print(new_distances)
	distances_a=sorted(new_distances, key=operator.itemgetter(0))
	#print('\nsorted:',distances_a)
	return distances_a

def predict_type(n,body_values,dorsal_values,type_values,test_x,test_y):
	type1_count = 0
	type0_count = 0
	distance_type = predict(test_x,test_y,body_values,dorsal_values,type_values)
	type_values = [row[1] for row in distance_type]
	#print('\n type_val:',type_values)
	type_values_n =type_values[0:n]
	#print('\ntype_values_n',type_values_n)
	type_values_n = np.array([type_values_n])
	#print('\ntype_values_n',type_values_n)
	type1_count = np.count_nonzero(type_values_n == 1.0)
	#print('type1_count',type1_count)
	type0_count = n-type1_count
	#print('type0_count',type0_count)
	if(type1_count > type0_count):
		predict_result = '1'
	elif(type1_count < type0_count):
		predict_result = '0'
	return predict_result


#pred_type = predict_type(3,body_values,dorsal_values,type_values,83.2,9)
#print('pred_type',pred_type)


def predict_result(n,body_values,dorsal_values,type_values,body_test,dorsal_test):
	predictions = list()
	for i in range(len(body_test)):
		predictions.append(predict_type(n,body_values,dorsal_values,type_values,body_test[i],dorsal_test[i]))
	print(predictions)
	return np.asarray(predictions)

pred_type_array = predict_result(3,body_values,dorsal_values,type_values,body_test,dorsal_test)


def confusion_matrix(pred_type_array,type_test):

	print('\n\n\n\n')
	error_count = 0
	matching_count = 0
	nonmatch_count = 0
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	accuracy = 0
	precision = 0
	accuracy_p = 0
	recall = 0
	print(pred_type_array,type_test)
	print("Confusion matrix for Type 0 :\n")
	for i in range(len(type_test)):
		print(i)
		if(float(pred_type_array[i]) == float(type_test[i])):
			print(i)
			matching_count = matching_count +1
			print('matching_count',matching_count)
			if((float(pred_type_array[i])==0) and (float(type_test[i])==0)):
				TP = TP+1
			elif((float(pred_type_array[i]==1)) and (float(type_test[i]==1))):
				TN = TN+1
		if(float(pred_type_array[i]) != float(type_test[i])):
			nonmatch_count = nonmatch_count +1
			if((float(pred_type_array[i])==0) and (float(type_test[i])==1)):
				FP = FP+1
			elif((float(pred_type_array[i])==1) and (float(type_test[i]==0))):
				FN = FN+1
	print('TP:',TP,'TN:',TN,'FP:',FP,'FN:',FN,'\n')
	print('\n\n\n\n')
	error_count = 0
	matching_count = 0
	nonmatch_count = 0
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	print("Confusion matrix for Type 1 :\n")
	for i in type_test:
		if(float(pred_type_array[i]) == float(type_test[i])):
			print('\nfloat(pred_type_array[i])',float(pred_type_array[i]))
			matching_count = matching_count +1
			print('matching_count',matching_count)
			if((float(pred_type_array[i])==1) and (float(type_test[i])==1)):
				TP = TP+1
			elif((float(pred_type_array[i])==0) and (float(type_test[i])==0)):
				TN = TN+1
		if(float(pred_type_array[i]) != float(type_test[i])):
			nonmatch_count = nonmatch_count +1
			if((float(pred_type_array[i])==1) and (float(type_test[i])==0)):
				FP = FP+1
			elif((float(pred_type_array[i])==0) and (float(type_test[i])==1)):
				FN = FN+1
	print('TP:',TP,'TN:',TN,'FP:',FP,'FN:',FN,'\n')
	error_count = FP+FN
	accuracy_p = (1-(error_count/int(tot*0.2)))*100
	accuracy = (TP+TN)/(TP+TN+FP+FN)
	#precision = (TP/(TP+FP))
	#recall = (TP/(TP+FN))
	return error_count,accuracy,accuracy_p,precision,recall

error_count,accuracy,accuracy_p,precision,recall = confusion_matrix(pred_type_array,type_test)
print(error_count,accuracy,accuracy_p,precision,recall)