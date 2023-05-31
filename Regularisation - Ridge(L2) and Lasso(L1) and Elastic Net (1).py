#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Using sklearn internal boston dataset from sklearn.datasets import load_boston 
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


house_sales = pd.read_csv('https://raw.githubusercontent.com/ammishra08/MachineLearning/master/Datasets/house_sales_data.csv') 
house_sales


# In[3]:


house_sales.columns


# #### Data Manipulation
#     * Handling Missing Values

# In[4]:


house_sales.isnull().sum()


# In[5]:


from pandas_profiling import ProfileReport


# In[6]:


ProfileReport(house_sales)


# #### Assumptions for Target Column
#     * Target Column(price) must be Normally Distributed

# In[7]:


from scipy.stats import norm
sns.set_style('whitegrid')
plt.figure(figsize = (12,9))
sns.distplot(house_sales['price'],fit = norm , color = 'crimson')
plt.show()


# In[8]:


from scipy import stats
sns.set_style('whitegrid')
fig = plt.figure()
res = stats.probplot(house_sales['price'], plot = plt)


# In[9]:


sns.set_style('whitegrid')
plt.figure(figsize = (10,5))
sns.boxplot('price', data = house_sales)


# #### Handling Outliers by Z- score

# In[10]:


house_sales.info()


# In[11]:


house_df = house_sales.drop(['id','date'], axis = 1)


# In[22]:


house_df.head()


# In[13]:


from scipy import stats
z = stats.zscore(house_df)


# In[14]:


np.where(np.abs(stats.zscore(house_df)) > 4)


# In[15]:


len(np.where(np.abs(stats.zscore(house_df)) > 4)[0]) / len(house_df)


# In[16]:


house_df.drop(np.where(np.abs(stats.zscore(house_df)) > 4)[0], axis = 0, inplace = True)


# In[17]:


from scipy.stats import norm
sns.set_style('whitegrid')
plt.figure(figsize = (11,8))
sns.distplot(house_df['price'], fit = norm, color = 'green')
plt.show()


# In[18]:


house_df.info()


# In[19]:


house_df.shape


# In[20]:


house_df.head(5)


# In[19]:


house_df.isnull().sum()


# #### Correlation

# In[20]:


plt.figure(figsize = (15,10))
sns.heatmap(house_df.corr(), annot = True, cmap = 'RdPu')


# In[21]:


plt.figure(figsize = (4,12))
sns.heatmap(house_df.corr()[['price']], annot = True, cmap = 'RdPu')


# In[22]:


house_df['waterfront'].value_counts()


# #### Split X and Y

# In[23]:


x_features = house_df.drop(['waterfront','yr_renovated','price'], axis = 1)


# In[24]:


y_target = house_df['price']


# In[25]:


# split data into train and test randomly
# choose a random state (0,11) 80% train and test = 20%
# random_state = (1,11) and keep random state value who give us higher accuracy
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_features, y_target, test_size = 0.2, random_state = 8)


# #### Data Preprocessing

# In[26]:


# MinMaxScaler - scale all the features with range (0,1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
# fit + transform
x_train_scaler = scaler.fit_transform(x_train)
# only transform for test
x_test_scaler = scaler.transform(x_test)


# #### Lasso Regression

# In[27]:


from sklearn.linear_model import Lasso
# hperparameter ; alpha = (0.00001-1), max_iter = higher if sample size is small
lasso_model = Lasso(alpha = 0.001, max_iter = 10000)


# In[28]:


lasso_model.fit(x_train_scaler, y_train)


# In[29]:


lasso_model.score(x_test_scaler, y_test)


# #### Regression Metrics

# In[30]:


# making new Predictions (yhat)
yhat = lasso_model.predict(x_test_scaler)


# In[31]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
# Here R2 is coefficient of Determination (0,1)

# R2 value tells us whether our model is good or bad if R2 value is closer to 1 then our model will be more accurate

# R2 (0,1): The more closer R squared is closer to 1 better is the performance

# The Proportion of variance in dependent variable(y) that is predictable from independent variables(x)

r2_score(y_test,yhat)


# In[32]:


mean_absolute_error(y_test,yhat)


# In[33]:


# MSE
mean_squared_error(y_test,yhat)


# In[34]:


# RMSE
np.sqrt(mean_squared_error(y_test,yhat))


# #### Features Selection using Lasso

# In[35]:


lasso_model.coef_


# In[36]:


lasso_coef = pd.DataFrame()
lasso_coef['columns'] = x_train.columns
lasso_coef['Coefficient Estimate'] = pd.Series(lasso_model.coef_)
print(lasso_coef)


# In[37]:


plt.figure(figsize = (20,9))
plt.bar(lasso_coef['columns'], lasso_coef['Coefficient Estimate'])
plt.show()


# #### Lasso Regression with different value of Alpha = 1

# In[38]:


from sklearn.linear_model import Lasso
# hperparameter ; alpha = (0.00001-1), max_iter = higher if sample size is small
lasso_model = Lasso(alpha = 1, max_iter = 100000)


# In[39]:


lasso_model.fit(x_train_scaler, y_train)


# In[40]:


lasso_model.score(x_test_scaler, y_test)


# In[41]:


lasso_coef = pd.DataFrame()
lasso_coef['columns'] = x_train.columns
lasso_coef['Coefficient Estimate'] = pd.Series(lasso_model.coef_)
print(lasso_coef)


# In[42]:


plt.figure(figsize = (20,9))
plt.bar(lasso_coef['columns'], lasso_coef['Coefficient Estimate'])
plt.show()


# #### Lasso Regression again after removal of columns who came when we change value of alpha = 1

# In[43]:


X = house_df.drop(['waterfront', 'sqft_basement','long','sqft_lot15','price','zipcode','yr_renovated'], axis = 1)
Y = house_df['price']


# In[44]:


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state = 8)


# #### Data Preprocessing Again

# In[45]:


scaler = MinMaxScaler(feature_range = (0,1))
# fit + transform
x_train_scaler = scaler.fit_transform(x_train)
# only transform for test
x_test_scaler = scaler.transform(x_test)


# In[46]:


lasso_model = Lasso(alpha = 1, max_iter = 100000)


# In[47]:


lasso_model.fit(x_train_scaler, y_train)


# In[48]:


lasso_model.score(x_test_scaler, y_test)


# #### Ridge Regression using X and Y split

# In[49]:


from sklearn.linear_model import Ridge
# hperparameter ; alpha = (0.00001-1), max_iter = higher if sample size is small
ridge_model = Ridge(alpha = 0.01, max_iter = 100000)


# In[50]:


ridge_model.fit(x_train_scaler, y_train)


# In[51]:


ridge_model.score(x_test_scaler , y_test)


# #### Regression Metrics for Ridge model

# In[52]:


# making new Predictions (yhat)
yhat = ridge_model.predict(x_test_scaler)


# In[53]:


# R2 value tells us whether our model is good or bad if R2 value is closer to 1 then our model will be more accurate

# R2 (0,1): The more closer R squared is closer to 1 better is the performance

# The Proportion of variance in dependent variable(y) that is predictable from independent variables(x)

r2_score(y_test,yhat)


# In[54]:


mean_absolute_error(y_test,yhat)


# In[55]:


# MSE
mean_squared_error(y_test,yhat)


# In[56]:


# RMSE
np.sqrt(mean_squared_error(y_test,yhat))


# In[ ]:




