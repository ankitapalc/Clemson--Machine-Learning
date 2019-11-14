# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:01:16 2019

@author: palan
"""

def printnum (n):
	for i in range(1,101):
		if(i%3 == 0 and i%15 != 0):
			print('Fizz')
		elif(i%5 == 0 and i%15 !=0):
			print('Buzz')
		elif(i%15 == 0):
			print('FizzBuzz')
		else:
			print(i)


