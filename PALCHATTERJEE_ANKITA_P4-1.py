# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 19:09:08 2019

@author: palan
"""
import os
import numpy as np

def cleantext(text):
	text = text.lower()
	text = text.strip()
	for letters in text:
		if letters in "`~!@#$%^&*()_-+|{}[]<>?/\=\"\':,;.0123456789":
			text = text.replace(letters," ")

	return text

def countwords(words, is_spam, counted):
	for each_word in words:
		if each_word in counted:
			if is_spam==1:
				counted[each_word][1] += 1
			else:
				counted[each_word][0] += 1
		else:
			if is_spam==1:
				counted[each_word] = [0,1]
			else:
				counted[each_word] = [1,0]
	#print('counted',counted)
	return counted

def make_percent_list(k, theCount, spam_count, ham_count):
	for each_key in theCount:
		theCount[each_key][0] = (theCount[each_key][0] + k) / (2*k+ham_count)
		theCount[each_key][1] = (theCount[each_key][1] + k) / (2*k+spam_count)
	return theCount

def conditional_prob(list_words,vocab,k):
	prob = 1
	#print('vocab',vocab)
	for v in vocab:
		for l in list_words:
			if(v==l):
				prob *= vocab[v][k]
			else:
				prob *= (1-vocab[v][k])
	#prob = np.log(prob)
	return prob

def calc_final(prob_s,prob_h,spam_prob,ham_prob):
	fprob = 0
	#fprob = (np.log(prob_s)*np.log(spam_prob))/((np.log(prob_s)*np.log(spam_prob))+((np.log(prob_h*ham_prob))))
	fprob = (prob_s*spam_prob)/((prob_s*spam_prob)+(prob_h*ham_prob))
	#print('fprob',fprob)
	if(fprob>0.5):
			result = 1
	else:
			result = 0
	return result

######################################################################
def confusion_matrix(pred_mail_type,actual_mail_type):
	nonmatch_count = 0
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	#print(pred_mail_type,actual_mail_type)
	print("Confusion matrix for Type Spam :\n")
	for i in range(len(pred_mail_type)):
		if(float(pred_mail_type[i]) == float(actual_mail_type[i])):
			if((float(pred_mail_type[i])==1) and (float(actual_mail_type[i])==1)):
				TP = TP+1
			elif((float(pred_mail_type[i])==0) and (float(actual_mail_type[i])==0)):
				TN = TN+1
		if(float(pred_mail_type[i]) != float(actual_mail_type[i])):
			nonmatch_count = nonmatch_count +1
			if((float(pred_mail_type[i])==1) and (float(actual_mail_type[i])==0)):
				FP = FP+1
			elif((float(pred_mail_type[i])==0) and (float(actual_mail_type[i])==1)):
				FN = FN+1
	print('TP:',TP,'TN:',TN,'FP:',FP,'FN:',FN,'\n')
	return TP,TN,FP,FN

def count_error_parameters(pred_mail_type,actual_mail_type):
	error_count = 0
	accuracy = 0
	precision = 0#.000001
	accuracy_p = 0
	recall = 0#.000001
	F1_score = 0#.000001

	TP,TN,FP,FN = confusion_matrix(pred_mail_type,actual_mail_type)

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

	print('\nNo of Spams(counted)',(TP),'\nNo of Hams(counted)',(TN))
	print('\nNo of Spams including errors(counted)',(TP+FP),'\nNo of Hams including errors(counted)',(TN+FN))
	print('\nerror_count',error_count,'\naccuracy',accuracy,'\naccuracy_p',accuracy_p,'\nprecision',precision,'\nrecall',recall,'\nF1_score',F1_score)
	return error_count,accuracy,accuracy_p,precision,recall,F1_score

######################################################################
spam_count =0
ham_count =0
spam_sub = list()
ham_sub = list()
counted = dict()
pred_mail_type = list()
prob_s = list()
prob_h = list()
actual_mail_type = list()
subject = list()
#vocab = dict()
# Importing the training dataset
dirpath = os.getcwd()
print("current directory is : " + dirpath)
filepath = " "
#If no filepath given as input
filepath = input("Enter file path for dataset(If the file is kept on run directory press Enter twice):")
if not filepath.strip():
	filepath = dirpath
#If no Filename given as input, Filename is fixed and provided
filename = input('Enter Training File Name : ')
if not filename.strip():
	filename = 'GEASTrain.txt'
stop_filename = input('Enter File Name for stop words: ')
if not stop_filename.strip():
	stop_filename = 'StopWords.txt'
filename = filepath+'\\'+filename
stop_filename = filepath+'\\'+stop_filename

with open(stop_filename, 'r', encoding='unicode-escape') as stop_filename:
	stop_file = [stop.rstrip('\n') for stop in stop_filename]


datafile = open(filename, 'r',encoding = 'unicode-escape')
textline = datafile.readline()
words = ''
while textline!= "":
	is_spam = int(textline[0])
	actual_mail_type.append(is_spam)
	textline = cleantext(textline[1:])
	subject.append(textline)
	if (is_spam==1):
		spam_count +=1
	else:
		ham_count +=1
	words = textline.split()
	words = set(words) - set(stop_file)
	counted = countwords(words,is_spam,counted)
	textline = datafile.readline()
print('\nNo of Spam emails in training file(actual)',spam_count,'\nNo of Ham emails in training file(actual)',ham_count)
vocab = (make_percent_list(1, counted, spam_count, ham_count))
spam_prob = spam_count/(spam_count+ham_count)
ham_prob = ham_count/(spam_count+ham_count)

for i in range(len(subject)):
	prob_s.append(conditional_prob(subject[i].split(),vocab,1))
	prob_h.append(conditional_prob(subject[i].split(),vocab,0))
	pred_mail_type.append(calc_final(prob_s[i],prob_h[i],spam_prob,ham_prob))

count_error_parameters(pred_mail_type,actual_mail_type)

#################################################################################
spam_sub_test = list()
ham_sub_test = list()
list_words_test = list()
counted_test = dict()
prob_s_test = list()
prob_h_test = list()
pred_mail_type_test = list()
actual_mail_type_test = list()
test_subject = list()
spam_count = 0
ham_count = 0
test_filename = input('\nEnter Testing File Name : ')
if not test_filename.strip():
	test_filename = 'GEASTest.txt'
test_filename = filepath+'\\'+test_filename
words = ''
datafile_test = open(test_filename, 'r',encoding = 'unicode-escape')
test_textline = datafile_test.readline()

while test_textline!= "":
	is_spam = int(test_textline[0])
	actual_mail_type_test.append(is_spam)
	if (is_spam==1):
		spam_count +=1
	else:
		ham_count +=1
	test_textline = cleantext(test_textline[1:])
	test_subject.append(test_textline)
	words = test_textline.split()
	words = set(words) - set(stop_file)
	test_textline = datafile_test.readline()
print('\nNo of Spam emails in test file(actual)',spam_count,'\nNo of Ham emails in test file(actual)',ham_count)
for i in range(len(test_subject)):
	prob_s_test.append(conditional_prob(test_subject[i].split(),vocab,1))
	prob_h_test.append(conditional_prob(test_subject[i].split(),vocab,0))
	pred_mail_type_test.append(calc_final(prob_s_test[i],prob_h_test[i],spam_prob,ham_prob))


count_error_parameters(pred_mail_type_test,actual_mail_type_test)

