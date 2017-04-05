import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as ax

def loadDataset():
    '''
        Dataset Column
        Gender (1=Male, 2=Female)
        Age Range ( 1=20-46, 2=46+)
        Head size (cm^3)
        Brain weight (grams)
    '''
    f1 = open('brainHead.csv','rb')
    x = []
    y = []
    i = 0
    data = f1.read()
    print(data)
    idata = data.split('\n')
    for i in xrange(len(idata)):
        element = idata[i].split(",")
        x.append(float(element[0]))
        temp = element[1]
        if i == len(idata):
            y.append(float(element[1]))
        else:
            y.append(float(temp[:-1]))
    print ("x = ",x)
    print ("y = ",y)
    print ()
    
if __name__ == "__main__":
    loadDataset()
    