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

labels = data['math score']
del data['math score']
# Check if there are any missing values
data.describe()

## Data Visualization - Helps get an idea of the problem at hand]
x_axis = ['Female', 'Male']
y_axis = data['gender'].value_counts()
print(y_axis)
plt.bar(x_axis, y_axis, color = "blue")
plt.show()

x_axis = ['group C', 'group D', 'group B', 'group E', 'group A']
y_axis = data['race/ethnicity'].value_counts()
print(y_axis)
plt.bar(x_axis, y_axis, color = "blue")
plt.show()

# You can make more of them to get a better idea but yeah that's the jist of it

## Making the training and testing datasets - We are using a 80% - 20% split here

train_data = data[:800]
test_data = data[800:]

train_labels = labels[:800]
test_labels = labels[800:]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_dim=7))
model.add(tf.keras.layers.Dense(64, input_shape=[1], activation='relu'))
model.add(tf.keras.layers.Dense(64, input_shape=[64], activation='relu'))
model.add(tf.keras.layers.Dense(1, input_shape=[64]))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

model.fit(train_data, train_labels, epochs=1000)
model.save('dnn_non_linear_regression_multi_var.h5')

model_output = []
output = model.predict(np.array(test_data))
index = 800

for elem in output:
    to_save = elem[0] - test_labels[index]
    index += 1
    model_output.append(to_save)

x_axis = np.linspace(1, 200, 200)
print(x_axis.size)

## Look at the y-axis labels and how the max differences decreased
plt.scatter(x_axis, model_output, color = "green")
plt.show()
