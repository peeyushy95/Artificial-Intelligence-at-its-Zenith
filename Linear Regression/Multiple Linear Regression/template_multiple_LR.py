# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('') 

#provide index of IV and DV
X = dataset.iloc[:,:].values
Y = dataset.iloc[:,:].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Encode Categorical data into Numerical values .. index of IV required
catg_ind = 1
labelencoder = LabelEncoder()
X[:, catg_ind] = labelencoder.fit_transform(X[:, catg_ind])
onehotencoder = OneHotEncoder(categorical_features = [catg_ind]) 
X = onehotencoder.fit_transform(X).toarray()

# Multiple Collinearity: Dummy Variable Problem
# Remember Count - 1 thing :) .. remove one column
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 69)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
# Perform Scaling operation 

# Backward Elimination
# Remove the variable with higher P values > .05 (general) one by one
import statsmodels.formula.api as sm

#append a column of ones (for b0)
X = np.append(arr = np.ones((len(X_train), 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)