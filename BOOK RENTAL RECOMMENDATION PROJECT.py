#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


userdf = pd.read_csv('BX-Users.csv',encoding = 'latin-1')
userdf


# In[3]:


userdf.head()


# In[4]:


# checking the null values
userdf.isnull().sum()


# In[5]:


# dropping the null values
userdf1 = userdf.dropna()
userdf1


# #### Import Book Data and Explore

# In[6]:


bookdf = pd.read_csv('BX-Books.csv', encoding='latin-1')


# In[7]:


bookdf


# In[8]:


bookdf.head()


# #### Import Book Rating Data and Explore

# In[9]:


bookratingdf = pd.read_csv('BX-Book-Ratings.csv',encoding='latin-1',nrows = 12000)


# In[10]:


bookratingdf


# In[11]:


bookratingdf.head()


# In[12]:


bookratingdf.describe()


# #### Merging the DataFrame

# In[13]:


df =  pd.merge(bookratingdf,bookdf,on='isbn')
df.head()


# #### Checking for unique user and unique books

# In[14]:


no_users = df.user_id.nunique()
no_books = df.isbn.nunique()


# In[15]:


no_users


# In[16]:


no_books


# In[17]:


df.info()


# #### Convert ISBN variable to numerical type

# In[18]:


isbn_list = df.isbn.unique()
print(" Length of isbn List:", len(isbn_list))
def get_isbn_numeric_id(isbn):
    #print ("  isbn is:" , isbn)
    itemindex = np.where(isbn_list==isbn)
    return itemindex[0][0]


# #### Convert user_id variable into numerical type

# In[19]:


userid_list = df.user_id.unique()
print(" Length of user_id List:", len(userid_list))
def get_user_id_numeric_id(user_id):
    #print ("  isbn is:" , isbn)
    itemindex = np.where(userid_list==user_id)
    return itemindex[0][0]


# #### Convert user_id and isbn variable into ordered list i.e from 0....n-1

# In[20]:


df['user_id_order'] = df['user_id'].apply(get_user_id_numeric_id)


# In[21]:


df['isbn_id'] = df['isbn'].apply(get_isbn_numeric_id)
df.head()


# In[22]:


df.head()


# In[23]:


df.info()


# #### Re-index columns to built Matrix

# In[24]:


new_col_order = ['user_id_order', 'isbn_id', 'rating', 'book_title', 'book_author','year_of_publication','publisher','isbn','user_id']
df = df.reindex(columns= new_col_order)
df.head()


# ## Train Test  and Split

# #### import train test split model

# In[25]:


from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size=0.30)


# ### Approach: You Will Use Memory-Based Collaborative Filtering

# Memory-Based Collaborative Filtering approaches can be divided into two main sections: user-item filtering and item-item filtering.
# 

# A user-item filtering will take a particular user, find users that are similar to that user based on similarity of ratings, and recommend items that those similar users liked.
# 
# 

# In contrast, item-item filtering will take an item, find users who liked that item, and find other items that those users or similar users also liked. It takes items as input and outputs other items as recommendations.

# * Item-Item Collaborative Filtering: “Users who liked this item also liked …”
# 

# * User-Item Collaborative Filtering: “Users who are similar to you also liked …”

# In both cases, you create a user-book matrix which is built from the entire dataset.

# Since you have split the data into testing and training, you will need to create two [828 x 8051] matrices (all users by all books). This is going to be a very large matrix.

# The training matrix contains 70% of the ratings and the testing matrix contains 30% of the ratings.

# #### Create two user-book matrix for training and testing

# In[26]:


#Create user-book matrix for training 
train_data_matrix = np.zeros((no_users, no_books))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]  
    
#Create user-book matrix for testing
test_data_matrix = np.zeros((no_users, no_books))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]


# #### Import Pairwise Model

# You can use the pairwise_distances function from sklearn to calculate the cosine similarity. Note, the output will range from 0 to 1 since the ratings are all positive.

# In[27]:


#Importing pairwise_distances function
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')


# In[28]:


user_similarity


# #### Make predictions

# In[29]:


#Defining custom function to make predictions
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
    return pred


# In[30]:


item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')


# #### Evaluation

# There are many evaluation metrics, but one of the most popular metric used to evaluate accuracy of predicted ratings is Root Mean Squared Error (RMSE).

# Since, you only want to consider predicted ratings that are in the test dataset, you filter out all other elements in the prediction matrix with: prediction[ground_truth.nonzero()].

# In[31]:


#Importing RMSE function 
from sklearn.metrics import mean_squared_error
from math import sqrt

#Defining custom function to filter out elements with ground_truth.nonzero
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


# #### Printing RMSE value for user based and item based collaborative filtering

# In[32]:


print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))


# ### Both the approach yield almost same result

# In[ ]:




