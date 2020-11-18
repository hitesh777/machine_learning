# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 18:29:48 2020

@author: Titans
"""
# The aim of this problem is to segment the clients of a wholesale distributor based on their 
# annual spending on diverse product categories, like milk, grocery, region, etc

# reading data:

import pandas as pd
cust_data = pd.read_csv(r"C:\Users\TITANS\Downloads\Wholesale customers data.csv")
cust_data.shape

# statistics of the data
x=cust_data.describe()
pd.DataFrame(x)

#checking missing values:
cust_data.info()


#Here, we see that there is a lot of variation in the magnitude of the data. 
#Variables like Channel and Region have low magnitude whereas 
#variables like Fresh, Milk, Grocery, etc. have a higher magnitude.

#Since K-Means is a distance-based algorithm, 
#this difference of magnitude can create a problem. 
#So let’s first bring all the variables to the same magnitude:




from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(cust_data)

# statistics of scaled data
x=pd.DataFrame(data_scaled).describe()


#The magnitude looks similar now..Next, let’s create a kmeans function and fit it on the data:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters=2, init='k-means++')

# fitting the k means algorithm on scaled data
y=kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)
pred=pd.DataFrame(pred)
pred.columns = ['pred_cluster']
new_ds= pd.concat([cust_data, pred], axis=1)
new_ds['pred_cluster'].value_counts()



## elbow curve

# fitting multiple k-means algorithms and storing the values in an empty list
SSE = []
   
for cluster in range(1,10):
    kmeans = KMeans( n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)


#cluster vs qty.
# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,10), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

# Looking at the above elbow curve, we can choose any number of clusters between 5 to 8. Let’s set the number of clusters as 6 and fit the model:

# k means using 5 clusters and k-means++ initialization
kmeans = KMeans( n_clusters = 6, init='k-means++')
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)
pred2 = kmeans.predict(data_scaled)

#let’s look at the value count of points in each of the above-formed clusters:

frame = pd.DataFrame(pred)
frame.columns = ['cluster_no']
new_ds= pd.concat([cust_data, frame], axis=1)

new_ds['cluster_no'].value_counts()
#profiling clusters

df1 = new_ds.query('(cluster_no == 0)')
df1['Channel'].value_counts()
df1['Region'].value_counts()
x=df1.describe()
profiling_df1 = pd.DataFrame(x)



df1 = new_ds.query('(cluster_no == 1)')
df1['Channel'].value_counts()
df1['Region'].value_counts()
x=df1.describe()
profiling_df1 = pd.DataFrame(x)
     




df1 = new_ds.query('(cluster_no == 2)')
df1['Channel'].value_counts()
df1['Region'].value_counts()
x=df1.describe()
profiling_df1 = pd.DataFrame(x)
     

df1 = new_ds.query('(cluster_no == 3)')
df1['Channel'].value_counts()
df1['Region'].value_counts()
x=df1.describe()
profiling_df1 = pd.DataFrame(x)




df1 = new_ds.query('(cluster_no == 4)')
df1['Channel'].value_counts()
df1['Region'].value_counts()
x=df1.describe()
profiling_df1 = pd.DataFrame(x)


df1 = new_ds.query('(cluster_no == 5)')
df1['Channel'].value_counts()
df1['Region'].value_counts()
x=df1.describe()
profiling_df1 = pd.DataFrame(x)




def dunn(X, labels) :
    n = labels.shape[0]
    groupes = np.unique(labels, return_counts=False)
    k = len(groupes)
    diam = []
    minInter = []

    for i in range(0, k) :
        maski = labels == groupes[i]
        diam.append(np.max(X[:, maski][maski, :]))
        for j in range(i+1, k) :
            maskj = labels == groupes[j]
            minInter.append(np.min(X[maski, :][:, maskj]))
    d = min(minInter) / max(diam)
    return d 


dun=dunn(data_scaled,)


from sklearn.cluster import k_means
from sklearn.metrics import pairwise_distances

from validclust import dunn


try:
    from sklearn.metrics import calinski_harabasz_score
except ImportError:
    
    from sklearn.metrics import calinski_harabaz_score
from jqmcvi import base     
    
from sklearn.datasets import load_iris
data = load_iris()['data']
_, labels, _ = k_means(cust_data, n_clusters=6)
#labels.value_counts()
dist = pairwise_distances(cust_data)
dunn(dist, labels)




