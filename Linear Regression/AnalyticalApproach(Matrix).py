import csv
import numpy as np
import matplotlib.pyplot as plt

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

class Linear_Regression :
    
    def __init__(self,dataset_x,dataset_y):
        self.x = dataset_x
        self.y = dataset_y
        
    def analytical_approach(self):    
        self.betaOne = 0.0
        self.betaZero = 0.0
        n = len(self.x)
        
        sum_x = sum(self.x)
        sum_y = sum(self.y)
        sum_xy = 0
        sum_x2 = 0
        for i in range(n):
            sum_x2 += self.x[i]*self.x[i]
        for i in range(n):
            sum_xy += self.x[i]*self.y[i]
        
        self.betaOne = (sum_xy - (1.0/n*sum_x*sum_y))/(sum_x2 - (1.0/n*(sum_x*sum_x)))
        self.betaZero = 1.0/n*(sum_y  - self.betaOne*sum_x)
           
    def R_function(self):
        self.predicted_y = []
        tempsum = 0.0
        n = len(self.x)
        for i in range(n):
            self.predicted_y.append(self.betaZero+self.betaOne*self.x[i])
            diff = self.y[i] - self.predicted_y[i]
            tempsum += diff*diff
        R = 1/(2.0*n)*tempsum
        print ("R = ",R)
        print ()
       
        
    def plot_regression_line(self):
        self.R_function()
        plt.scatter(self.x,self.y)
        plt.plot(self.x,self.predicted_y, '--r')
        plt.title('Plot Regression Line')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()  
        
    
if __name__ == "__main__":
    
    dataset_x = []
    dataset_y = []
    load_dataset(dataset_x,dataset_y)
    plot_dataset(dataset_x,dataset_y)
    
    lr = Linear_Regression(dataset_x,dataset_y)
    lr.analytical_approach()
    lr.plot_regression_line()  
   
    