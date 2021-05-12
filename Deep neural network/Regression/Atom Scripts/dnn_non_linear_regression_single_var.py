## Import the libraries needed to successfully execute the code
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

## Problem Statement - In this example we will be predicting the math scores using the dataset we have been given

## Get the data
data = pd.read_csv(r'D:\DATA_BASE\student_performance\StudentsPerformance.csv')
data.head()

## Data preprocessing

# Tokenizing columns (converting into numbers to feed our DNN) - We will use label encoding here
# Note that in this case one hot encoding is better but we will see it in a later example

data['gender'] = data['gender'].replace(['male', 'female'], [0, 1])
data['race/ethnicity'] = data['race/ethnicity'].replace(['group A', 'group B', 'group C', 'group D', 'group E'], [0, 1, 2, 3, 4])
data['parental level of education'] = data['parental level of education'].replace(["some high school", "bachelor's degree", "some college", "master's degree", "high school", "associate's degree"], [0, 1, 2, 3, 4, 5])
data['lunch'] = data['lunch'].replace(["standard", "free/reduced"], [0, 1])
data['test preparation course'] = data['test preparation course'].replace(["completed", "none"], [0, 1])

# Normalize reading and writing scores
data['reading score'] = data['reading score'].div(100.0)
data['writing score'] = data['writing score'].div(100.0)
data.head()

## Data Visualization - Helps get an idea of the problem at hand]
plt.plot(data['reading score'], data['math score'], 'o', color = "blue")
plt.ylabel('Math score')
plt.xlabel('Reading score')
plt.title('Relationship with reading score')
plt.show()

labels = data['math score']
del data['math score']
data = data['reading score']
# Check if there are any missing values
data.describe()

## Making the training and testing datasets - We are using a 80% - 20% split here

train_data = data[:800]
test_data = data[800:]

train_labels = labels[:800]
test_labels = labels[800:]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=[1]))
model.add(tf.keras.layers.Dense(64, input_shape=[1], activation='relu'))
model.add(tf.keras.layers.Dense(64, input_shape=[64], activation='relu'))
model.add(tf.keras.layers.Dense(1, input_shape=[64]))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

model.fit(train_data, train_labels, epochs=1000)
model.save('dnn_non_linear_regression_single_var.h5')

## Let's see how it performed on the set
x = np.linspace(0, 1, 101)
y = model.predict(x)
data = pd.read_csv(r'D:\DATA_BASE\student_performance\StudentsPerformance.csv')
data['reading score'] = data['reading score'].div(100.0)
plt.plot(data['reading score'], data['math score'], 'o', color = "blue")
plt.plot(x, y, color = "red")
plt.show()
