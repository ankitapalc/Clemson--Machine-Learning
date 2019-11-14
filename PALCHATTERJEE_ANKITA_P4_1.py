# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:45:48 2019

@author: palan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

################################################################
def read_file(f_name):
	# Importing the training dataset
	dirpath = os.getcwd()
	print("current directory is : " + dirpath)
	filepath = " "

	#If no filepath given as input
	filepath = input("Enter file path for training dataset(If the file is kept on run directory press Enter twice):")
	if not filepath.strip():
		filepath = dirpath
	#If no Filename given as input, Filename is fixed and provided
	filename = input('Enter Training File Name : ')
	if not filename.strip():
		filename = f_name
	stop_filename = input('Enter File Name for stop words: ')
	if not stop_filename.strip():
		stop_filename = 'StopWords.txt'

	filename = filepath+"\\"+filename
	stop_filename = filepath+"\\"+stop_filename

	with open(filename, "r", encoding="latin1") as datafile:
		string_file = [line.rstrip('\n') for line in datafile]
	with open(stop_filename, "r", encoding="latin1") as stop_filename:
		stop_file = [stop.rstrip('\n') for stop in stop_filename]
	return string_file,stop_file
###Remove punctuations
def remove_punct(string_file):
	punctuationset = '''!@#$%^&*()_+=-[]{};:'"\,<>./?~|`'''
	# remove punctuation from the string
	no_punct_df = ' '
	for letter in string_file:
		if letter not in punctuationset:
			no_punct_df = no_punct_df + letter
	return no_punct_df
###Split the file into words
def bag_of_words(string_file,stop_file):
	string = ''
	stop_string = ''
	string = string.join(string_file)
	stop_string = ' '.join(stop_file)
	words = string.split()
	stop_words = stop_string.split()
	wordset = set(words)-set(stop_words)
	unique_list = (list(wordset))
	return unique_list
###create bag of words
def remove_dup(lines):
	words = np.unique(lines)
	return words

def cnt_vec(bag_of_word,naive_file):
	cnt_ham = 0
	cnt_spam = 0
	for w in bag_of_word:
		if (naive_file[i][0] == 0):
			cnt_ham = cnt_ham+1
		else:
			cnt_spam = cnt_spam+1


################################################################

naive_file,stop_file = read_file('GEASTrain.txt')
for i in range(len(naive_file)):
	naive_file[i] = remove_punct(naive_file[i])
bag_of_word = bag_of_words(naive_file,stop_file)


