# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 22:31:13 2017

@author: Megamindo_0
"""
__author__ = 'Peeyush Yadav'

import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_dataset(xy,x,y):
    
    with open('chessboard.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[0] != 'A':
                xy.append([float(row[0]),float(row[1]),int(row[2])])
                x.append([float(row[0]),float(row[1])])
                y.append(int(row[2]))
    
   
#    print ("x = ",x)
#    print ("y = ",y)
#    print ()

def plot_dataset(xy,x,y):
    A0 = [row[0] for row in xy if row[2] == 0]
    A1 = [row[0] for row in xy if row[2] == 1]
    B0 = [row[1] for row in xy if row[2] == 0]
    B1 = [row[1] for row in xy if row[2] == 1]
    plot0 = plt.scatter(A0,B0, marker='+', color = 'red')
    plot1 = plt.scatter(A1,B1, marker = 'o', color = 'green')
    plt.legend((plot0, plot1), ('label 0', 'label 1'), scatterpoints = 1)
    plt.title("Scatter Plot")
    plt.xlabel('A')
    plt.ylabel('B')
    plt.show()  
'''
class Support_Vector_Machine :
    
    def __init__(self,dataset_x,dataset_y):
        self.x = dataset_x
        self.y = dataset_y
        
    def linear_svm(self):
         
    def plot_regression_line(self):
        self.R_function()
        plt.scatter(self.x,self.y)
        plt.plot(self.x,self.predicted_y, '--r')
        plt.title('Plot Regression Line')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()  
    
    def plot_cost_function(self):
'''        
    
if __name__ == "__main__":
    dataset=[]
    dataset_x = []
    dataset_y = []
    load_dataset(dataset,dataset_x,dataset_y)
    plot_dataset(dataset,dataset_x,dataset_y)
    '''
    svm = Support_Vector_Machine(dataset_x,dataset_y)
    svm.linear_svm()
    '''
   
    