# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

 1.Import the necessary packages using import statement.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Import KMeans and use for loop to cluster the data.

4.Predict the cluster and plot data graphs.

5.Print the outputs and end the program
 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Roop Sagar S L
RegisterNumber:  212223040175
*/

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data=pd.read_csv('/content/Mall_Customers_EX8 (1).csv')
data
X=data[['Annual Income (k$)','Spending Score (1-100)']]
X
plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
k=5
kmeans =KMeans(n_clusters=k)
kmeans.fit(X)
centroids=kmeans.cluster_centers_
labels=kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
colors=['r','g','b','c','m']
for i in range(k):
  cluster_points=X[labels==i]
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],
                             color=colors[i],label=f'Cluster {i+1}')
  distances=euclidean_distances(cluster_points,[centroids[i]])
  radius=np.max(distances)
  circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,color='k',label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

```


## Output:

![1](https://github.com/user-attachments/assets/f13dbfd6-25b2-47f9-a0f2-f2e0ea898879)

![2](https://github.com/user-attachments/assets/6f8bd6f7-f803-448a-9829-c54f873e3a92)

![3](https://github.com/user-attachments/assets/a6ee0f11-3a88-4537-9ab1-ac459ad21701)

![4](https://github.com/user-attachments/assets/a56c1a7d-d8c5-44c5-ad9e-fb60a031f492)

![5](https://github.com/user-attachments/assets/89b24bcd-0444-4659-b93e-b23c00b27040)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
