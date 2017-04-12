import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_dataset(x,y):
    
    with open('data.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            x.append(float(row[0]))
            y.append(float(row[1]))
    
   
#    print ("x = ",x)
#    print ("y = ",y)
#    print ()

def plot_dataset(x,y):
    
    plt.scatter(x,y)
    plt.title('Scatter plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()  

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
        
    
if __name__ == "__main__":
    
    dataset_x = []
    dataset_y = []
    load_dataset(dataset_x,dataset_y)
    plot_dataset(dataset_x,dataset_y)
    
    svm = Support_Vector_Machine(dataset_x,dataset_y)
    svm.linear_svm()
   
    