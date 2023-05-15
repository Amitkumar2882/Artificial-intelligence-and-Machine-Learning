#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
# Using sklearn internal boston dataset from sklearn.datasets import load_boston
from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings('ignore')


# In[2]:


boston = load_boston()
boston


# In[3]:


boston_df = pd.DataFrame(boston.data, columns = boston.feature_names)
boston_df


# In[4]:


boston_df['MEDV'] = boston.target


# In[5]:


boston_df


# #### Data Manipulation¶
#     * Handling Missing Values

# In[6]:


from pandas_profiling import ProfileReport


# In[7]:


ProfileReport(boston_df)


# #### Features and Target
#     * we create x feature and y target

# In[8]:


# Features
X = boston_df.drop(['CHAS','MEDV'], axis = 1)
# Target
Y = boston_df['MEDV']


# In[9]:


X


# In[10]:


Y


# #### Train & Test Spliting Data

# In[11]:


# split data into train and test randomly
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state = 0)


# In[12]:


x_train


# In[13]:


x_test


# In[14]:


y_test.shape


# #### Data Preprocessing
# 
# * There are two techniques to reduce down impact of high magnitude;
# 
# 
# * a)Standard Scaler
# 
# 
# * b) Min Max Scaler

# In[15]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)


# In[16]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(2, activation = 'relu'))  # Hidden layer with 2 nodes(2 Artificial neuron)
model.add(Dense(1,activation = 'relu' ))  # Output layer

#compile model
model.compile(loss = 'mse' , optimizer = 'adam', metrics = ['mae'])

from livelossplot import PlotLossesKerasTF


# In[17]:


# Train model
result = model.fit(x_train_scaler,y_train,epochs = 500,callbacks = [PlotLossesKerasTF()],
           validation_data = (x_test_scaler,y_test))


# #### Increasing the Number of Hidden Layers

# In[22]:


model = Sequential()
model.add(Dense(15, activation = 'relu'))  # Hidden layer with 2 nodes(2 Artificial neuron)
model.add(Dense(15, activation = 'relu'))
model.add(Dense(1,activation = 'relu' ))  # Output layer

#compile model
model.compile(loss = 'mse' , optimizer = 'adam', metrics = ['mae'])

from livelossplot import PlotLossesKerasTF


# In[23]:


# Train model
result = model.fit(x_train_scaler,y_train,epochs = 50,callbacks = [PlotLossesKerasTF()],
           validation_data = (x_test_scaler,y_test))


# In[ ]:


model.predict(x_test_scaler)


# In[ ]:


y_test.head(30)


# In[ ]:


x_new = [[9.23230,0.0,18.10,0.631,6.216,100.0,1.1691,24.0,666.0,20.2,366.15,9.53]]


# In[ ]:


X_new_scaler = scaler.transform(x_new)


# In[ ]:


model.predict(X_new_scaler)


# In[ ]:




