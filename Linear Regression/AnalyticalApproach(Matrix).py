import csv
import numpy as np
import matplotlib.pyplot as plt

def load_dataset(x,y):
    
    with open('data.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            x.append([1, float(row[0])])
            y.append([float(row[1])])
    
   
#    print ("x = ",x)
#    print ("y = ",y)
#    print ()

def plot_dataset(x,y):
    
    plt.scatter([row[1] for row in x ],y)
    plt.title('Scatter plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()  

class Linear_Regression :
    
    def __init__(self,dataset_x,dataset_y):
        self.x = dataset_x
        self.y = dataset_y
    
        
    def analytical_approach(self):
        self.scale_data()
        xMatrix = np.matrix(self.x)
        yMatrix = np.matrix(self.y)
        xMatrixT = np.transpose(xMatrix)
        beta = (((xMatrixT.dot(xMatrix)).getI()).dot(xMatrixT)).dot(yMatrix)
        self.betaOne = float(beta[1])
        self.betaZero = float(beta[0])
        print("Beta: ",self.betaZero,self.betaOne)    
   
        
    def scale_data(self):
        x = np.array(self.x)
        y = np.array(self.y)
        meanX = x[:,1].mean()
        meanY = y[:,0].mean()
        stdX =  x[:,1].std()
        stdY =  y[:,0].std()
        
        for ind in range(len(x)):
            self.x[ind][1] = (self.x[ind][1] - meanX)/stdX
            self.y[ind][0] = (self.y[ind][0] - meanY)/stdY


    def R_function(self):
        self.predicted_y = []
        tempsum = 0.0
        n = len(self.x)
        for i in range(n):
            self.predicted_y.append(self.betaZero+self.betaOne*self.x[i][1])
            diff = self.y[i][0] - self.predicted_y[i]
            tempsum += diff*diff
        R = 1/(2.0*n)*tempsum
        print ("R = ",R)
        print ()
       
        
    def plot_regression_line(self):
        self.R_function()
        plt.scatter([row[1] for row in self.x ],self.y)
        plt.plot([row[1] for row in self.x ],self.predicted_y, '--r')
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
   
    