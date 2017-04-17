"""
Created on Mon Apr 18 00:23:01 2017

@author: Megamindo_0
"""
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
        self.scale_data()
    
    def scale_data(self):
        x = np.array(self.x)
        y = np.array(self.y)
        meanX = np.mean(x)
        meanY = np.mean(y)
        stdX =  np.std(x)
        stdY =  np.std(y)
        
        for ind in range(len(x)):
            self.x[ind] = (self.x[ind] - meanX)/stdX
            self.y[ind] = (self.y[ind] - meanY)/stdY
        
    def batch_gradient_descent(self):
        self.alpha = .001 
        self.betaZero = 0.0
        self.betaOne = 0.0
        self.betaTwo = 0.0
        iter = 0
        ep = .0001
        max_iter = 10000
        n = len(self.x)
        converged = False
        # J(theta)
        J = 2.0/n * sum([(self.betaZero + self.betaOne*self.x[i] + self.betaTwo*(self.x[i]**2) - self.y[i]) 
                    for i in range(n)])
    
        while not converged :          
            gradZero = 1.0/n*sum([(self.betaZero + self.betaOne*self.x[i] + self.betaTwo*(self.x[i]**2)- self.y[i]) 
                                  for i in range(n)])
            gradOne = 1.0/n*sum([(self.betaZero + self.betaOne*self.x[i] + self.betaTwo*(self.x[i]**2) - self.y[i])*self.x[i] 
                                  for i in range(n)])
            gradTwo = 1.0/n*sum([(self.betaZero + self.betaOne*self.x[i] + self.betaTwo*(self.x[i]**2) - self.y[i])*(self.x[i]**2) 
                                  for i in range(n)])
            self.betaZero = self.betaZero - self.alpha * gradZero
            self.betaOne = self.betaOne - self.alpha * gradOne
            self.betaTwo = self.betaTwo - self.alpha * gradTwo
            
            error = 2.0/n * sum([(self.betaZero + self.betaOne*self.x[i] + self.betaTwo*(self.x[i]**2) - self.y[i]) 
                                for i in range(n)])
            
            '''
            if abs(J - error) < ep :
                print ("Converged")
                converged = True 
            '''
            J = error           
            iter += 1           
            if iter == max_iter :
                print("Max iteration Reached")
                converged = True
        print(self.betaZero, self.betaOne, self.betaTwo)
    
    def R_function(self):
        self.predicted_y = []
        tempsum = 0
        n = len(self.x)
        for i in range(n):
            self.predicted_y.append(self.betaZero+self.betaOne*self.x[i] + self.betaTwo*(self.x[i]**2))
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
    
if __name__ == "__main__":
    
    dataset_x = []
    dataset_y = []
    load_dataset(dataset_x,dataset_y)
    plot_dataset(dataset_x,dataset_y)
    
    lr = Linear_Regression(dataset_x,dataset_y)
    lr.batch_gradient_descent()
    lr.plot_regression_line()  
   
    