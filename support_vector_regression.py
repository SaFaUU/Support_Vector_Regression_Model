#Data Preprocessing
#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv("Position_Salaries.csv")

#Independent Variable Matrix/ Vector
X = dataset.iloc[:,1:2].values

#Making Dependent Variable Matrix/ Vector
y= dataset.iloc[:, 2].values

#Splitting the dataset into Training and Test set
'''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''

#Feature Scaling
y= np.array(y).reshape(-1,1)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#Fitting SVR to the dataset
#create regressor
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

#Predicting Single Value/ new result with regression
y_pred = sc_y.inverse_transform(np.array(regressor.predict(sc_X.transform((np.array(6.5).reshape(1,-1))))).reshape(1,-1))


#Visualising the SVR Results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='purple')
plt.title("Salary vs Levels (SVR)")
plt.xlabel("Levels")
plt.ylabel("Salary")
plt.show()