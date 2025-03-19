#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# ### Data Collection & Analysis

# In[2]:


# Loading the data from csv file to a pandas DataFrame

customer_data = pd.read_csv(r'B:\Machine Learning Projects Datasets\Customer Segmentation\Mall_Customers.csv')


# In[3]:


# Finding the number of rows & columns
customer_data.shape


# In[4]:


# First 5 rows in the Dataframe

customer_data.head()


# In[5]:


# Getting some informations about the dataset
customer_data.info()


# In[6]:


# checking for missing values
customer_data.isnull().sum()


# ### Choosing the Annual income column & Spending Score column

# In[7]:


X = customer_data.iloc[:,[3,4]].values


# In[8]:


print(X)


# ### Choosing the number of clusters
# 
# - WCSS = Within Clusters Sum of Squares

# In[9]:


# Finding WCSS value for diffrent number of clusters


wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    
    wcss.append(kmeans.inertia_)


# In[10]:


# Plot an elbow graph


sns.set()
plt.plot(range(1,11),wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.show()


# ### Optimum Number of Clusters = 5
# 
# - training the K-Means Clustering Model

# In[11]:


kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)


# Ruturn a label for each data point based on their cluster

Y = kmeans.fit_predict(X)

print(Y)


# ### Visualizing all the Clusters

# In[12]:


# Plotting all the clusters adn their centroids

plt.figure(figsize = (8,8))
plt.scatter(X[Y == 0,0], X[Y == 0,1], s=50, c='green', label='cluster 1')
plt.scatter(X[Y == 1,0], X[Y == 1,1], s=50, c='red', label='cluster 2')
plt.scatter(X[Y == 2,0], X[Y == 2,1], s=50, c='yellow', label='cluster 3')
plt.scatter(X[Y == 3,0], X[Y == 3,1], s=50, c='violet', label='cluster 4')
plt.scatter(X[Y == 4,0], X[Y == 4,1], s=50, c='blue', label='cluster 5')



# Plot the Centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 100, c = 'cyan', label= 'Centroids')


plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()


# In[ ]:




