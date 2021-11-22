import numpy as np
import pandas as pd

# Loading the data
data = pd.read_csv('iris.data', header=None)

# Creating csv file from the data
data.rename(columns={0: 'sepal_length', 1: 'sepal_width', 2: 'petal_length', 3: 'petal_width', 4: 'class'}, inplace=True)
data.to_csv('iris_data.csv', index=False)

# Visualizing the csv file
data_df = pd.read_csv('iris_data.csv')
print(data_df.head())

# Splitting data into x and y(label)
x = data.iloc[:, 0: 4].values
y = data.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
encoded_y = LabelEncoder()
y = encoded_y.fit_transform(y)

# Splitting the dataset into Training set and Test set (Cross Validation)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Training model on train data
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

# Testing model on test data
prediction = model.predict(x_test)

# Getting the weights and bias
w = np.array(model.coef_)
b = model.intercept_
print('---------Weights and Bias----------')
print('weights:', w)
print('bias:', b)

# Getting the training score and test score for the model
print('----------Train & Test Accuracy-----------')
print('Train accuracy:', model.score(x_train, y_train))
print('Test accuracy:', model.score(x_test, y_test))

# Evaluating the model
from sklearn import metrics
mse = metrics.mean_squared_error(y_test, prediction)
mae = metrics.mean_absolute_error(y_test, prediction)
r2_score = metrics.r2_score(y_test, prediction)
print('---------Evaluation-----------')
print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)
print('R2 score:', r2_score)
