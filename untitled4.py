# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 20:52:57 2019

@author: palan
"""

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

#Importing input file
#If no path provided the file could be used from current directory path
dirpath = os.getcwd()
print("current directory is : " + dirpath)
filepath = " "
#filepath = r"C:\Users\palan\OneDrive\Desktop\Clemson- Machine Learning"
filepath = input("Enter file path(If the file is placed on run directory press Enter twice):")
#Assigning filepath to current directory path
if not filepath.strip():
    filepath = dirpath
#Filename is fixed and provided
filename = "FF62.txt"
#open file and start reading
datafile1=open(filepath+"\\"+filename)
print(datafile1)
datafile= pd.read_csv(datafile1,header = None, sep='\t')
tot= int(datafile.loc[0,0])