"""
Created on Mon Apr 18 00:11:31 2017

@author: Megamindo_0
"""

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

class Linear_Regression :
    
    def __init__(self,dataset_x,dataset_y):
        self.x = dataset_x
        self.y = dataset_y
        
    def stochastic_gradient_descent(self):
        self.alpha = .0001       
        self.betaOne = 0.0
        self.betaZero = 0.0
        iter = 0
        ep = .0001
        max_iter = 1000
        n = len(self.x)
        converged = False
        # J(theta)
        J = 1.0/(2*n) * sum([(self.betaZero + self.betaOne*self.x[i] - self.y[i]) for i in range(n)])
    
        while not converged : 
            for i in range(n):
                gradZero = 1.0/n *(self.betaZero + self.betaOne*self.x[i] - self.y[i]) 
                gradOne = 1.0/n * (self.betaZero + self.betaOne*self.x[i] - self.y[i])*self.x[i]                
                self.betaZero = self.betaZero - self.alpha * gradZero
                self.betaOne = self.betaOne - self.alpha * gradOne            
            error = 1.0/(2*n) * sum([(self.betaZero + self.betaOne*self.x[i] - self.y[i]) for i in range(n)])
            
            if abs(J - error) < ep :
                print ("Converged")
                converged = True           
            J = error           
            iter += 1           
            if iter == max_iter :
                print("Max iteration Reached")
                converged = True
        print(self.betaZero,self.betaOne)
    
    def R_function(self):
        self.predicted_y = []
        tempsum = 0
        n = len(self.x)
        for i in range(n):
            self.predicted_y.append(self.betaZero+self.betaOne*self.x[i])
            diff = self.y[i] - self.predicted_y[i]
            tempsum += diff*diff
        R = 1/(2.0*n)*tempsum
        print ("R = ",R)
        print("")
        
    def plot_regression_line(self):
        self.R_function()
        plt.scatter(self.x,self.y)
        plt.plot(self.x,self.predicted_y, '--r')
        plt.title('Plot Regression Line')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()  
    
    def plot_cost_function(self):
        costfn = []
        iteration = 50
        betaZero = np.linspace(0.0, 0.1, num = iteration)
        betaOne = np.linspace(1.0, 1.5, num = iteration)
        betaZeroEntries = []
        betaOneEntries = []
        n = len(self.x)
        for i in range(iteration):
            for k in range(iteration):
                del  self.predicted_y[:]
                tempsum = 0.0
                betaZeroEntries.append(betaZero[i])
                betaOneEntries.append(betaOne[k])            
                for j in range(n):
                    self.predicted_y.append(betaZero[i]+ betaOne[k]*self.x[j])
                    diff = self.y[j] - self.predicted_y[j] 
                    tempsum += diff*diff
                costfn.append(1/(2.0*n)*tempsum)
    
        fig = plt.figure()
        figure = fig.add_subplot(1,1,1, projection="3d")
        figure.plot(betaZeroEntries, betaOneEntries, costfn, linestyle = "none", marker = ".", mfc = "none", markeredgecolor = "green")
        figure.set_title("CostFunction")
        figure.set_xlabel("BetaZero")
        figure.set_ylabel("BetaOne")
        figure.set_zlabel("Cost")
        plt.show()
    
if __name__ == "__main__":
    
    dataset_x = []
    dataset_y = []
    load_dataset(dataset_x,dataset_y)
    plot_dataset(dataset_x,dataset_y)
    
    lr = Linear_Regression(dataset_x,dataset_y)
    lr.stochastic_gradient_descent()
    lr.plot_regression_line()   
    lr.plot_cost_function()
   
    