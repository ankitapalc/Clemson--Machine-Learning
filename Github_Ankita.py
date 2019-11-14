# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 22:44:33 2019

@author: palan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
import operator
import os
import csv
import random


#If no path provided the file could be used from current directory path
dirpath = os.getcwd()
print("current directory is : " + dirpath)
#n=int(input('Enter Number of nearest Neighour:'))
#Kin = int(input('Enter Number of Folds::'))
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


def splitDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'r') as file_name:

		lines = csv.reader(file_name,delimiter='\t')
		next(lines)
		dataset = list(lines)
		print('len(dataset)',len(dataset))
		for x in range(len(dataset)-1):
			for y in range(2):

				dataset[x][y] = float(dataset[x][y])
				if random.random() < int(tot*0.33):
					trainingSet.append(dataset[x])
				else:
					testSet.append(dataset[x])

def euclideanDistance(X1, X2, length):
	distance = 0
	for x in range(length):
		distance += pow((X1[x] - X2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = [],neighbors = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))

	for i in range(k):
		neighbors.append(distances[i][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for i in range(len(neighbors)):
		response = neighbors[i][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]



def main():
	# prepare data
	trainingSet=[]
	testSet=[]

	splitDataset(filename, tot*.33, trainingSet, testSet)
	print ('Training set: ' + repr(len(trainingSet)))
	print ('Testing set: ' + repr(len(testSet)))
	# generate predictions
	predictions=[]
	k = 3
	for i in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[i], k)
		result = getResponse(neighbors)
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	#accuracy = getAccuracy(testSet, predictions)
	#print('Accuracy: ' + repr(accuracy) + '%')

main()


