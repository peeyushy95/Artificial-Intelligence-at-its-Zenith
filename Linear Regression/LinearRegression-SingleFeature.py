import csv
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as ax

def load_dataset(x,y):
    '''
        Dataset Column
        Gender (1=Male, 2=Female)
        Age Range ( 1=20-46, 2=46+)
        Head size (cm^3)
        Brain weight (grams)
    '''
    
    with open('brainHead.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            x.append(int(row[2]))
            y.append(int(row[3]))
    
   
#    print ("x = ",x)
#    print ("y = ",y)
#    print ()

def plot_dataset(x,y):
    
    plt.scatter(x,y)
    plt.title('Scatter plot')
    plt.xlabel('Head Size')
    plt.ylabel('Brain Weight')
    plt.show()  

class Linear_Regression :
    
    def __init__(self,dataset_x,dataset_y):
        self.x = dataset_x
        self.x = dataset_y
        
    def gradient_descent(x,y):
        self.betaOne = 0
        self.betaTwo = 0
        
    
if __name__ == "__main__":
    dataset_x = []
    dataset_y = []
    load_dataset(dataset_x,dataset_y)
    plot_dataset(dataset_x,dataset_y)
    
    lr = Linear_Regression(dataset_x,dataset_y)
    lr.gradient_descent()
   
    