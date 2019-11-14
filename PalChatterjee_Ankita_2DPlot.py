import matplotlib.pyplot as plt
import numpy as np
import os

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
filename = "IrisData.txt"
#open file and start reading
iris_csv = open(filepath+"\\"+filename)
data_file=np.genfromtxt(iris_csv,dtype='|U50',delimiter = '\t')
#fetch the columns and store them in lists
sepal_len = data_file[:,0]
petal_len = data_file[:,2]
flower_type = data_file[:,4]
#get the unique types of flowers in file
uniq_fl_type = np.unique(flower_type)
#Plotting figure using Matplotlib
fig = plt.figure()
ax = plt.axes()
fig.set_size_inches(18,15)
#for different flower types plot points with different colors
for i in range(len(flower_type)):
    if(flower_type[i] == uniq_fl_type[0]):
        plt.scatter(float(sepal_len[i]), float(petal_len[i]),color='red',marker='D')
    elif(flower_type[i] == uniq_fl_type[1]):
        plt.scatter(float(sepal_len[i]), float(petal_len[i]),color='green',marker='s')
    elif(flower_type[i] == uniq_fl_type[2]):
        plt.scatter(float(sepal_len[i]), float(petal_len[i]),color='blue',marker='o')
#labeling x axis and y axis
plt.title("Iris Flower Plot",weight="bold", size="xx-large")
plt.xlabel("Sepal Length(in cm)---->",weight="normal", size="large")
plt.ylabel("Petal Length(in cm)---->",weight="normal", size="large")
plt.show()
#saving the plot as png file
fig.savefig(filepath+'/PALCHATTERJEE_ANKITA_MyPlot.png')
#closing the file we were reading from
iris_csv.close()