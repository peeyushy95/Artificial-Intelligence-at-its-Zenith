import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataSet = pd.read_csv('Data.csv')
X = dataSet.iloc[:,1:2].values
Y = dataSet.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X, Y)

plt.scatter(X, Y, color='cyan', label='Data Set')
X_grid = np.arange(min(X), max(X), .01).reshape(-1,1)
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue', label='Regressor')
plt.show()