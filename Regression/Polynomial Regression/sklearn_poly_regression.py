import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_set = pd.read_csv('Data.csv')
X = data_set.iloc[:,1:2].values
Y = data_set.iloc[:,2].values

#adding polynomical features
from sklearn.preprocessing import PolynomialFeatures
poly_regr = PolynomialFeatures(degree = 4)
X_poly = poly_regr.fit_transform(X)

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X_poly, Y)

plt.scatter(X, Y, color='red', label="Data Points")
plt.plot(X,linear_reg.predict(X_poly), color='green', label="Prediction")
plt.legend()
plt.show()