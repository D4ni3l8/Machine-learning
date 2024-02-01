import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

print(df.head())

# mean of total production grouped by each year
prod_per_year = df.groupby('year').totalprod.mean().reset_index()

print(prod_per_year)

x = prod_per_year["year"] #column of year in the prod_per_year dataframe
X = x.values.reshape(-1, 1) #reshaping the dataframe

y = prod_per_year["totalprod"]

plt.scatter(X, y)
plt.show()

regr = linear_model.LinearRegression() # linear regression model
regr.fit(X, y)
print(regr.coef_)
print(regr.intercept_)

y_predict = regr.predict(X)
plt.plot(X, y_predict) # creating line of best fit
plt.show()

X_future = np.array(range(2013, 2050)) # 
print(X_future)
X_future = X_future.reshape(-1, 1) # reshapning the array 
future_predict = regr.predict(X_future) #prediciting values of X_future

plt.plot(X_future, future_predict)
plt.show() # displaying the predicition
