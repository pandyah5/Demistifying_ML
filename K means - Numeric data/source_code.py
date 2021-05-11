from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r'D:\DATA_BASE\Mall_customers\Mall_Customers.csv')
data.head()

num_cluster = 2
colors = ['red', 'blue']
x_axis1 = data['Age']
y_axis1 = data['Annual Income (k$)']

fig, axs = plt.subplots(2, 3)

fig.set_figheight(15)
fig.set_figwidth(15)

X1 = []
for index in range(0, len(x_axis1)):
    X1.append([x_axis1[index], y_axis1[index]])

axs[0, 0].plot(x_axis1, y_axis1, 'o')

x_axis2 = data['Annual Income (k$)']
y_axis2 = data['Spending Score (1-100)']

X2 = []
for index in range(0, len(x_axis2)):
    X2.append([x_axis2[index], y_axis2[index]])

axs[0, 1].plot(x_axis2, y_axis2, 'o')

x_axis3 = data['Spending Score (1-100)']
y_axis3 = data['Age']

X3 = []
for index in range(0, len(x_axis3)):
    X3.append([x_axis3[index], y_axis3[index]])

axs[0, 2].plot(x_axis3, y_axis3, 'o')

kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(X1)
cat1 = kmeans.labels_

kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(X2)
cat2 = kmeans.labels_

kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(X3)
cat3 = kmeans.labels_

for i in range(0, len(X1)):
    axs[1, 0].plot(X1[i][0], X1[i][1], 'o', color = str(colors[cat1[i]]))
    axs[1, 1].plot(X2[i][0], X2[i][1], 'o', color = str(colors[cat2[i]]))
    axs[1, 2].plot(X3[i][0], X3[i][1], 'o', color = str(colors[cat3[i]]))

plt.show()
