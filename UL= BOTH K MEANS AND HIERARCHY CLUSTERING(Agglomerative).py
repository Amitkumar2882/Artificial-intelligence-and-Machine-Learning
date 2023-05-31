#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv("zoo.csv")


# In[2]:


data.head()


# In[3]:


data.info()


# In[4]:


data[['animal_name', 'class_type']]


# In[5]:


data['animal_name'].unique()


# In[6]:


data['class_type'].unique()


# In[7]:


import numpy as np
labels = data['class_type']
print(np.unique(labels.values))

from matplotlib import pyplot as plot
get_ipython().run_line_magic('matplotlib', 'inline')
fig, ax = plot.subplots()
(labels.value_counts()).plot(ax=ax, kind='bar')


# In[8]:


data.shape


# In[9]:


features=data.values[:,1:-1]
features.shape


# In[54]:


from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances


# In[11]:


obj = AgglomerativeClustering(n_clusters= 7 , affinity= 'euclidean', linkage= 'average')


# In[12]:


obj.fit(features)


# In[13]:


y_pred = obj.fit_predict(features)
y_pred


# In[14]:


newdata = data.copy()
newdata['label'] = y_pred + 1


# In[15]:


newdata.head()


# In[16]:


newdata['label'].unique()


# In[17]:


newdata['class_type'].unique()


# In[18]:


data['ecu'] = y_pred


# In[19]:


data['ecu'] = data['ecu']+1


# In[20]:


data.head()


# In[21]:


obj = AgglomerativeClustering(n_clusters= 7 , affinity= 'euclidean', linkage= 'ward')
y_pred = obj.fit_predict(features)
data['cos'] = y_pred
data['cos'] = data['cos'] + 1
data


# In[ ]:





# In[ ]:





# In[22]:


model = AgglomerativeClustering(n_clusters=7,linkage="average", affinity="cosine")


# In[23]:


model.fit(features)


# In[24]:


model.labels_


# In[25]:


#print(np.unique(model.labels_))


# In[26]:


#labels = labels -1


# In[27]:


#from sklearn.metrics import mean_squared_error


# In[28]:


#score = mean_squared_error(labels,model.labels_)


# In[29]:


#abs_error = np.sqrt(score)
#print(abs_error)


# In[ ]:





# In[30]:


## Adding Dendogram:


# In[31]:


import matplotlib.pyplot as plt
X = features


# In[32]:


import scipy.cluster.hierarchy as sch


# In[33]:


dendogram = sch.dendrogram(sch.linkage(X, method='ward'))


# In[34]:


# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'average'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


#     These are routines for agglomerative clustering.
# 
#     linkage(y[, method, metric, optimal_ordering]) -  Perform hierarchical/agglomerative clustering.
#     single(y)  --  Perform single/min/nearest linkage on the condensed distance matrix y.
#     complete(y) -- Perform complete/max/farthest point linkage on a condensed distance matrix.
#     average(y) --   Perform average/UPGMA linkage on a condensed distance matrix.
#     weighted(y) --  Perform weighted/WPGMA linkage on the condensed distance matrix.
#     centroid(y) --  Perform centroid/UPGMC linkage.
#     median(y) --    Perform median/WPGMC linkage.
#     ward(y) --     Perform Ward’s linkage on a condensed distance matrix.

# In[35]:


### Optimal Method


#        Form flat clusters from the hierarchical clustering defined by the given linkage matrix.
#        fcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None)
#     
#         Z -  The hierarchical clustering encoded with the matrix returned by the linkage function.
#         t - For criteria ‘inconsistent’, ‘distance’ or ‘monocrit’, this is the threshold to apply when forming flat clusters
#         The criterion to use in forming flat clusters. This can be any of the following values:
#                       inconsistent - If a cluster node and all its descendants have an inconsistent value less than or equal 
#                       to t, then all its leaf descendants belong to the same flat cluster. 
#                       
#                       distance - Forms flat clusters so that the original observations in each flat cluster have no 
#                       greater a cophenetic distance than t
#                       
#                       monocrit - Finds a minimum threshold r so that the cophenetic distance between any two 
#                       original observations in the same flat cluster is no more than r and no more than t flat
#                       clusters are formed.
#         depth - The maximum depth to perform the inconsistency calculation. It has no meaning for the other criteria. 
#                 Default is 2
#                 
#         R - The inconsistency matrix to use for the ‘inconsistent’ criterion. This matrix is computed if not provided.
#                 

# In[36]:


from scipy.cluster.hierarchy import fcluster


# In[37]:


from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(X, 'ward', metric='euclidean')
Z.shape


# In[38]:


max_d = 10


# In[39]:


clusters = fcluster(Z, max_d, criterion='distance')
clusters


# In[40]:


#from scipy.cluster.hierarchy import single, cophenet
#from scipy.spatial.distance import pdist


# In[41]:


#c = cophenet(Z)


# In[42]:


#from scipy.cluster.hierarchy import cophenet
#from scipy.spatial.distance import pdist

#c, coph_dists = cophenet(Z, pdist(X))


# In[43]:


plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=40,  # show only the last p merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()


# In[44]:


wcss = []
for cluster in range(1,11):
    kmeans = KMeans(n_clusters= cluster, init= 'k-means++', random_state=123)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[45]:


wcss


# In[46]:


plt.plot(range(1,11), wcss)


# In[ ]:





# In[47]:


obj = AgglomerativeClustering(n_clusters= 4 , affinity= 'euclidean', linkage= 'ward')
y_pred = obj.fit_predict(features)
data['c4'] = y_pred
data['c4'] = data['c4'] + 1
data


# In[48]:


cluster1 = data[data['c4'] ==1]


# In[49]:


cluster2 = data[data['c4'] ==2]


# In[50]:


cluster1['animal_name'].unique()


# In[51]:


cluster2['animal_name'].unique()


# In[52]:


cluster1['animal_name'].value_counts()


# In[53]:


cluster2['animal_name'].value_counts()


# In[ ]:





# In[ ]:





# In[ ]:




