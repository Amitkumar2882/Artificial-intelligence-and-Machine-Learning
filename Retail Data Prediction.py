#!/usr/bin/env python
# coding: utf-8

# # Retail - Sales Prediction of More than 1000 Stores

# ### DESCRIPTION
# 
# * Demand Forecast is one of the key tasks in Supply Chain and Retail Domain in general. It is key in effective operation and       optimization of retail supply chain.
#   
# 
# * Training Data Description: Historic sales at Store-Day level for about two years for a retail giant, for more than 1000         stores. Also, other sale influencers like, whether on a particular day the store was fully open or closed for renovation,       holiday and special event details, are also provided.
#     
# 
# * we are using Machine Learning and Deep Learning Techniques for the Prediction of Sales.we are using Linear
#   Regression,Regularisation,Ensemble Techniques and Time Series Techniques for Prediction.
#       
# 
# * Deep Learning we are using LSTM Technique with fine tunning Hyper Parameter like Batch_size, epochs, optimizer and               Learning rate and other important parameters.

# #### Import important Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Import Dataset
Retail_df = pd.read_csv('retail_train_data.csv')
Retail_df.Date = pd.to_datetime(Retail_df.Date)
Retail_df


# In[3]:


Retail_df.info()


# ## Data Wrangling

# In[4]:


# Looking for missing values
Retail_df.isnull().sum()


# In[5]:


# replace 'a', 'b', 'c' with 1
Retail_df['StateHoliday'].replace(['a','b','c'], ['1','2','3'],regex = True, inplace = True)
Retail_df['StateHoliday']


# In[6]:


Retail_df['StateHoliday'] = Retail_df['StateHoliday'].astype('int64')


# In[7]:


Retail_df['StateHoliday'].value_counts()


# In[8]:


Retail_df.info()


# #### Assumptions for Target Column
#     * Target Column(Sales) must be Normally Distributed

# In[9]:


from scipy.stats import norm
sns.set_style('whitegrid')
plt.figure(figsize = (12,8))
sns.distplot(Retail_df['Sales'],fit = norm , color = 'green')
plt.show()


# In[10]:


from scipy import stats
sns.set_style('whitegrid')
fig = plt.figure()
res = stats.probplot(Retail_df['Sales'], plot = plt)


# #### Outliers

# In[11]:


sns.set_style('whitegrid')
plt.figure(figsize = (12,6))
sns.boxplot( x = 'Sales', data = Retail_df , palette = 'plasma')


# #### Handling Outlier by Z - score

# In[12]:


new_df = Retail_df.drop(['Date'], axis = 1)


# In[13]:


new_df.shape


# In[14]:


new_df.head()


# In[15]:


from scipy import stats
z = stats.zscore(new_df)


# In[16]:


np.where(np.abs(stats.zscore(new_df)) > 4)


# In[17]:


len(np.where(np.abs(stats.zscore(new_df)) > 4)[0]) / len(new_df)*100


# In[18]:


new_df.drop(np.where(np.abs(stats.zscore(new_df)) > 4)[0], axis = 0, inplace = True)


# In[19]:


from scipy.stats import norm
sns.set_style('whitegrid')
plt.figure(figsize = (11,8))
sns.distplot(new_df['Sales'], fit = norm, color = 'green')
plt.show()


# In[20]:


from scipy import stats
sns.set_style('whitegrid')
fig = plt.figure()
res = stats.probplot(new_df['Sales'], plot = plt)


# ## Exploratory Data Analysis 

# In[21]:


fig, axs = plt.subplots(2,3, figsize=(15,12))
fig.subplots_adjust(hspace=0.4)
axs=axs.ravel()
A_list = ['Open', 'Promo','StateHoliday', 'SchoolHoliday','DayOfWeek']
i=0
for col in A_list:
    sns.scatterplot(new_df.Customers,new_df.Sales,hue=new_df[col],ax=axs[i])
    axs[i].set_title(col)
    i+=1
 
    fig.show()


# From the above EDA analysis we can conclude as follow:
# 
# * Sales is zero when shop is closed.
# 
# * Sales is high when promo codes and discount is available.
# 
# * Sales is either very low or very high on State Holidays.
# 
# * Sales is high when schools are open.

# ## Linear Regression single model for all stores, using storeId as a feature.
# 

# #### Correlation between Features

# In[22]:


plt.figure(figsize = (15,10))
sns.heatmap(new_df.corr() , annot = True , cmap = 'magma')
plt.show()


# #### Features and Target
#     *  create x Features and y target 
#     

# In[23]:


# Features
X_feature = np.array(new_df.drop(['Sales'], axis = 1))
# Target
Y_target = np.array(new_df['Sales'])


# ### Random Sampling

# #### Train and Spilt of Data 

# In[24]:


# split data into train and test randomly
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_feature, Y_target, test_size = 0.2, random_state = 100)


# #### Data Preprocessing

# * Scaling Down the Features Range to reduce impact of high Magnitude values in columns
#     
#     
#     * Min Max Scaler

# In[25]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler(feature_range = (0,1))
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)


# #### Linear Regression Model

# In[26]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()


# In[27]:


# training the lr model
lin_reg.fit(x_train_scaler, y_train)


# In[28]:


# Test Score - R2 Value this is R squared value whose range (0,1)
print('Training score', lin_reg.score(x_train_scaler, y_train))
print('Test Score ' , lin_reg.score(x_test_scaler, y_test))


# #### Regression Metrics

# In[29]:


# this is predicted output
y_pred = lin_reg.predict(x_test_scaler)
y_pred


# In[30]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
r2_score(y_test,y_pred)


# In[31]:


def error_cal(y_true,y_pred):
    RMSE = np.sqrt(mean_squared_error(y_true,y_pred))
    MAE = mean_absolute_error(y_true,y_pred)
    return RMSE,MAE


# In[32]:


all_store_lr = error_cal(y_test,y_pred)
all_store_lr


# #### MAE

# In[33]:


mean_absolute_error(y_test,y_pred)


# #### MSE

# In[34]:


# MSE
mean_squared_error(y_test,y_pred)


# #### RMSE

# In[35]:


# RMSE
np.sqrt(mean_squared_error(y_test,y_pred))


# In[36]:


plt.figure(figsize=(16,8))
plt.plot(y_pred[:100],label = 'y_predicted')
plt.plot(y_test[:100], label = 'y_true')
plt.legend()
plt.show()


# #### Prediction of Hidden Data Set

# In[37]:


hidden_data = pd.read_csv('test_data_hidden.csv')


# In[38]:


hidden_data


# In[39]:


new_features = hidden_data.drop(['Date','Sales'], axis = 1)
new_target = hidden_data['Sales']


# In[40]:


new_features


# In[41]:


new_target.shape


# In[42]:


new_features_scaler = scaler.transform(new_features)


# In[43]:


lin_reg.predict(new_features_scaler)


# In[44]:


new_pred = pd.DataFrame({ 'Predict_values' : lin_reg.predict(new_features_scaler)})


# In[45]:


new_pred


# In[46]:


Actual = pd.DataFrame({'Actual_values' : new_target})


# In[47]:


result = pd.concat([Actual, new_pred], axis = 1)


# In[48]:


result


# ## Regularization - Ridge Model for all stores, using storeId as a feature.

# In[49]:


from sklearn.linear_model import Ridge
model = Ridge(alpha = 1, max_iter = 10000 , random_state = 4, normalize = False)


# In[50]:


model.fit(x_train_scaler, y_train)


# In[51]:


model.score(x_test_scaler, y_test)


# #### Regression Metrics for Ridge model

# In[52]:


# making new Predictions (y_pred)
y_pred = model.predict(x_test_scaler)


# In[53]:


r2_score(y_test,y_pred)


# #### MAE

# In[54]:


mean_absolute_error(y_test,y_pred)


# #### MSE

# In[55]:


# MSE
mean_squared_error(y_test,y_pred)


# #### RMSE

# In[56]:


# RMSE
np.sqrt(mean_squared_error(y_test,y_pred))


# #### Comparison between Linear Regression Model and Ridge Model

# * There is no such difference between Linear Regression and Ridge Regression.
# 
# * RMSE, MAE AND MSE values of both model are almost equal

# ## Regularization - Elastic Net for all stores, using storeId as a feature.

# In[57]:


from sklearn.linear_model import ElasticNet


# In[58]:


model = ElasticNet(alpha = 0.001, max_iter = 100000, l1_ratio = 0.5)


# In[59]:


model.fit(x_train_scaler, y_train)


# In[60]:


model.score(x_test_scaler, y_test)


# #### Regression Metrics for Elastic Net model

# In[61]:


# making new Predictions (y_pred)
y_pred = model.predict(x_test_scaler)


# In[62]:


r2_score(y_test,y_pred)


# #### MAE

# In[63]:


mean_absolute_error(y_test,y_pred)


# #### MSE

# In[64]:


# MSE
mean_squared_error(y_test,y_pred)


# #### RMSE

# In[65]:


# RMSE
np.sqrt(mean_squared_error(y_test,y_pred))


# ## Ensemble Technique- Random Forest Regressor 

# In[66]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()


# In[67]:


model = RandomForestRegressor(n_estimators = 50, max_depth = 30, criterion='mse')


# In[68]:


model.fit(x_train_scaler, y_train)


# In[69]:


y_pred = model.predict(x_test_scaler)


# In[70]:


print('Training Accuracy', model.score(x_train_scaler, y_train))


# In[71]:


print('Test Accuracy' , model.score(x_test_scaler, y_test))


# In[72]:


r2_score(y_test,y_pred)


# #### Regression Metrics for Random Forest Regressor model

# #### MAE

# In[73]:


mean_absolute_error(y_test,y_pred)


# #### MSE

# In[74]:


# MSE
mean_squared_error(y_test,y_pred)


# #### RMSE

# In[75]:


# RMSE
np.sqrt(mean_squared_error(y_test,y_pred))


# ## Extra Tree Regressor Model

# In[76]:


from sklearn.ensemble import ExtraTreesRegressor


# In[77]:


model = ExtraTreesRegressor(n_estimators = 100, max_depth = 20, criterion='mse')


# In[78]:


model.fit(x_train_scaler, y_train)


# In[79]:


print('Training Accuracy', model.score(x_train_scaler, y_train))


# In[80]:


print('Test Accuracy' , model.score(x_test_scaler, y_test))


# ### Comparison between all above Model

# * Randam Forest Regressor Have high test accuracy 97% 
# 
# 
# * Extra Tree Regressor Model have test accuracy 90%
# 
# 
# * Elastic Net Model have test accuracy 85%
# 
# 
# * Ridge Model have test accuracy 85%
# 
# 
# * Linear Regression Model test accuracy 84%
# 
# 
# * All above model are not overfit/underfit 

# ## Linear Regression Model for each Store

# In[81]:


# Features
x = np.array(new_df.drop(['Sales'], axis = 1))
# Target
y = np.array(new_df['Sales'])


# In[82]:


def model_single_store(x,y):
    lin_reg = LinearRegression(normalize=True)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)
    lin_reg.fit(x_train,y_train)
    y_pred = lin_reg.predict(x_test)
    return y_test,y_pred


# In[83]:


stores = [1,2,3,4,5,6,7]
RMSE_array_lr = []
MAE_array_lr=[]
for store in range(1,8):
    data = new_df[new_df.Store==store]
    data.drop('Store',axis=1,inplace=True)
    y=np.array(data['Sales'])
    x=np.array(data.drop('Sales',axis=1))
    y_true,y_pred = model_single_store(x,y)
    RMSE_1,MAE_1 = error_cal(y_true,y_pred)
    RMSE_array_lr.append(RMSE_1)
    MAE_array_lr.append(MAE_1)


# In[84]:


error_lr = pd.DataFrame()
error_lr['Stores'] = stores
error_lr['RMSE'] = RMSE_array_lr
error_lr['MAE'] = MAE_array_lr
error_lr


# In[85]:


plt.figure(figsize=(16,8))
N = 8
x = np.arange(1,N)
plt.bar(x,height=error_lr.RMSE,label = 'RMSE',width = 0.3, color = 'g')
plt.bar(x+0.3,height=error_lr.MAE,label = 'MAE',width = 0.3, color = 'r')
plt.xticks(stores)
plt.legend()
plt.title('Error Output of Linear regerssion model for single store')
plt.show()


# #### c) Which performs better and Why?

# * Above model shows that accuracy increases when we train our model for individual store, because Sales might also depend on geographical or locality of the store, So when we predict for individual store then this factor could be treated as constant and can be neglected.

# #### d) Try Ensemble of b) and c). What are the findings?

# In[86]:


from sklearn.ensemble import AdaBoostRegressor


# In[87]:


def adaboost_single_store(x,y):
    lr = AdaBoostRegressor(n_estimators=200, learning_rate= 1)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)
    lr.fit(x_train,y_train)
    y_pred = lr.predict(x_test)
    return y_test,y_pred


# In[88]:


RMSE_array_ada = []
MAE_array_ada =[]
for store in range(1,8):
    data = new_df[new_df.Store==store]
    data.drop('Store',axis=1,inplace=True)
    y=np.array(data['Sales'])
    x=np.array(data.drop('Sales',axis=1))
    y_true,y_pred = adaboost_single_store(x,y)
    RMSE_1,MAE_1 = error_cal(y_true,y_pred)
    RMSE_array_ada.append(RMSE_1)
    MAE_array_ada.append(MAE_1)


# In[89]:


error_output_ada = pd.DataFrame()
error_output_ada['Stores'] = stores
error_output_ada['RMSE'] = RMSE_array_ada
error_output_ada['MAE'] = MAE_array_ada
error_output_ada


# In[90]:


plt.figure(figsize=(16,8))
N= 8
x=np.arange(1,N)
plt.bar(x,height=error_lr.RMSE,label = 'Linear Regression',width=0.3, color = 'y')
plt.bar(x+0.3,height=error_output_ada.RMSE,label = 'Ada Boost Regression',width=0.3, color = 'b')
plt.xticks(stores)
plt.legend()
plt.title('RMSE of Linear Regression vs Ada-Boost Regression')
plt.show()


# In[91]:


plt.figure(figsize=(16,8))
N=8
x=np.arange(1,N)
plt.bar(x,height=error_lr.MAE,label = 'Linear Regression',width=0.3)
plt.bar(x+0.3,height=error_output_ada.MAE,label = 'Ada Boost Regression',width=0.3)
plt.xticks(stores)
plt.legend()
plt.title('MAE of Linear Regression vs Ada-Boost Regression')
plt.show()


# * By comparing RMSE and MAE of ada-boost regression and Linear regression, we can say that there is not much difference between these models

# ### e) Use Regularized Regression. It should perform better in an unseen test set. Any insights??

# ##### Ridge Regression Model for each store

# In[92]:


def ridge_single_store(x,y):
    ridge = Ridge(alpha=0.00001,normalize=True, max_iter = 10000, random_state = 42 )
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)
    ridge.fit(x_train,y_train)
    y_pred = ridge.predict(x_test)
    return y_test,y_pred


# In[93]:


RMSE_array_rdg = []
MAE_array_rdg =[]
for store in range(1,8):
    data = new_df[new_df.Store==store]
    data.drop('Store',axis=1,inplace=True)
    y=np.array(data['Sales'])
    x=np.array(data.drop('Sales',axis=1))
    y_true,y_pred = ridge_single_store(x,y)
    RMSE_1,MAE_1 = error_cal(y_true,y_pred)
    RMSE_array_rdg.append(RMSE_1)
    MAE_array_rdg.append(MAE_1)


# In[94]:


error_output_rdg = pd.DataFrame()
error_output_rdg['Stores'] = stores
error_output_rdg['RMSE'] = RMSE_array_rdg
error_output_rdg['MAE'] = MAE_array_rdg
error_output_rdg


# ### f) Open-ended modeling to get possible predictions.

# ##### Random Forest Regressor Model

# In[95]:


def random_forest(x,y):
    rdm = RandomForestRegressor(n_estimators=120,criterion='mse',max_depth = 30)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)
    rdm.fit(x_train,y_train)
    y_pred = rdm.predict(x_test)
    return y_test,y_pred


# In[96]:


RMSE_array_rdm = []
MAE_array_rdm =[]
for store in range(1,8):
    data = new_df[new_df.Store==store]
    data.drop('Store',axis=1,inplace=True)
    y=np.array(data['Sales'])
    x=np.array(data.drop('Sales',axis=1))
    y_true,y_pred = random_forest(x,y)
    RMSE_1,MAE_1 = error_cal(y_true,y_pred)
    RMSE_array_rdm.append(RMSE_1)
    MAE_array_rdm.append(MAE_1)


# In[97]:


error_output_rdm = pd.DataFrame()
error_output_rdm['Stores'] = stores
error_output_rdm['RMSE'] = RMSE_array_rdm
error_output_rdm['MAE'] = MAE_array_rdm
error_output_rdm


# In[98]:


plt.figure(figsize=(16,8))
N=8
x=np.arange(1,N)
plt.bar(x,height=error_lr.RMSE,label = 'Linear Regression',width=0.2)
plt.bar(x+0.2,height=error_output_ada.RMSE,label = 'Ada Boost Regression',width=0.2)
plt.bar(x+0.4,height=error_output_rdg.RMSE,label = 'Ridge Regression',width=0.2)
plt.bar(x+0.6,height=error_output_rdm.RMSE,label = 'Random forest',width=0.2)
plt.xticks((2*x+0.6)/2,stores)
plt.xlabel('Store ID')
plt.legend()
plt.title('RMSE of Linear Regression vs Ada-Boost Regression')
plt.show()


# In[99]:


print('Average RMSE Linear regression Error: {}'.format(error_lr.RMSE.mean()))
print('Average RMSE Ada-boost regression Error: {}'.format(error_output_ada.RMSE.mean()))
print('Average RMSE Ridge regression Error: {}'.format(error_output_rdg.RMSE.mean()))
print('Average RMSE Random Forest regression Error: {}'.format(error_output_rdm.RMSE.mean()))


# * From Here we can conclude that upto this point random forest performs best.

# ### Other Regression Techniques:

# #### 1. When store is closed, sales = 0. Can this insight be used for Data Cleaning? Perform this and retrain the model. Any benefitsof this step?

# * When store is closed then there will be no sale. Hence remove that rows.

# In[100]:


open_store = new_df[new_df.Open == 1]
open_store.drop('Open',axis=1,inplace=True)


# In[101]:


open_store.head()


# In[102]:


RMSE_array_lrc = []
MAE_array_lrc=[]
for store in range(1,8):
    data = open_store[open_store.Store==store]
    data.drop('Store',axis=1,inplace=True)
    y=np.array(data['Sales'])
    x=np.array(data.drop('Sales',axis=1))
    y_true,y_pred = model_single_store(x,y)
    RMSE_1,MAE_1 = error_cal(y_true,y_pred)
    RMSE_array_lrc.append(RMSE_1)
    MAE_array_lrc.append(MAE_1)


# In[103]:


error_output_lrc = pd.DataFrame()
error_output_lrc['Stores'] = stores
error_output_lrc['RMSE'] = RMSE_array_lrc
error_output_lrc['MAE'] = MAE_array_lrc
error_output_lrc


# In[104]:


fig, axs = plt.subplots(1,2, figsize=(15,6))
fig.subplots_adjust(hspace=0.4)
axs=axs.ravel()
N=8
x=np.arange(1,N)
i=0
for col in ['RMSE','MAE']:
    axs[i].bar(x,height=error_lr[col],label = 'Linear Regression',width=0.2)
    axs[i].bar(x+0.2,height=error_output_lrc[col],label = 'Linear Regression with assumption\nNo sale when shop is closed)',width=0.2)
    axs[i].legend()
    axs[i].set_title(col+' Error')
    i+=1


# * The above graph shoes that both types of error get increased when we removed the rows when store are closed.

# #### 2. Use Non-Linear Regressors like Random Forest or other Tree-based Regressors.

# ##### a) Train a single model for all stores, where storeId can be a feature.

# In[105]:


open_store


# In[106]:


y=np.array(data['Sales'])
x=np.array(data.drop('Sales',axis=1))
y_true,y_pred = random_forest(x,y)


# In[107]:


RMSE_rdm,MAE_rdm = error_cal(y_true,y_pred)


# In[108]:


print('Root mean squared error: ',RMSE_rdm)
print('Mean absolute error: ',MAE_rdm)


# In[109]:


plt.figure(figsize=(16,8))
plt.plot(y_pred[:100],label = 'Sales forecast')
plt.plot(y_true[:100],label = 'Actual Sales')
plt.legend()
plt.title('Random Forest Regression taking all stores')
plt.show()


# ##### b) Train separate models for each store.

# * Dimensional Reduction techniques like, PCA and Tree’s Hyperparameter Tuning will be required. Cross-validate to find the best parameters. Infer the performance of both the models

# In[110]:


from sklearn.decomposition import PCA


# In[111]:


open_store.reset_index(drop=True,inplace=True)


# In[112]:


open_store


# In[113]:


x = open_store.drop(['Sales','Store'],axis=1)
x = StandardScaler().fit_transform(x)


# In[114]:


pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)


# In[115]:


pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)


# In[116]:


principalDf = pd.DataFrame(data = principalComponents
 , columns = ['PC_1', 'pc_2','PC_3'])


# In[117]:


finaldf = pd.concat([open_store[['Store','Sales']],principalDf],axis=1)


# In[118]:


finaldf.head()


# In[119]:


finaldf.reset_index(drop=True,inplace=True)


# In[120]:


def get_stats(model,x_train,y_train,x_test,y_test):
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    RMSE,MAE = error_cal(y_test,y_pred)
    return [RMSE,MAE]


# In[121]:


from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()


# In[122]:


y = np.array(finaldf['Sales'])
x = np.array(finaldf.drop('Sales',axis=1))


# In[123]:


score_rdm = []
score_dt = []
kf = StratifiedKFold(n_splits=5)
for train_index,test_index in kf.split(x,y):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=53)
    score_dt.append(get_stats(DecisionTreeRegressor(),x_train,y_train,x_test,y_test))
    score_rdm.append(get_stats(RandomForestRegressor(n_estimators=10),x_train,y_train,x_test,y_test))


# In[124]:


k_fold_df = pd.DataFrame()
k_fold_df['decision_tree']=pd.DataFrame(score_dt,columns=['RMSE','MAE']).mean(axis=0)
k_fold_df['Random_forest']=pd.DataFrame(score_rdm,columns=['RMSE','MAE']).mean(axis=0)


# In[125]:


k_fold_df


# In[126]:


# for individual stores
## random forest model
RMSE_array_rdm_pca = []
MAE_array_rdm_pca = []
for store in range(1,8):
    data = finaldf[finaldf.Store==store]
    data.drop('Store',axis=1,inplace=True)
    y=np.array(data['Sales'])
    x=np.array(data.drop('Sales',axis=1))
    y_true,y_pred = random_forest(x,y)
    RMSE_1,MAE_1 = error_cal(y_true,y_pred)
    RMSE_array_rdm_pca.append(RMSE_1)
    MAE_array_rdm_pca.append(MAE_1)


# In[127]:


error_output_rdm_pca = pd.DataFrame()
error_output_rdm_pca['Stores'] = stores
error_output_rdm_pca['RMSE'] = RMSE_array_rdm_pca
error_output_rdm_pca['MAE'] = MAE_array_rdm_pca
error_output_rdm_pca


# In[128]:


# decision tree model
def decision_tree(x,y):
    model = DecisionTreeRegressor()
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=36)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    return y_test,y_pred


# In[129]:


## decision tree model
RMSE_array_dt_pca = []
MAE_array_dt_pca = []
for store in range(1,8):
    data = finaldf[finaldf.Store==store]
    data.drop('Store',axis=1,inplace=True)
    y=np.array(data['Sales'])
    x=np.array(data.drop('Sales',axis=1))
    y_true,y_pred = decision_tree(x,y)
    RMSE_1,MAE_1 = error_cal(y_true,y_pred)
    RMSE_array_dt_pca.append(RMSE_1)
    MAE_array_dt_pca.append(MAE_1)


# In[130]:


error_output_dt_pca = pd.DataFrame()
error_output_dt_pca['Stores'] = stores
error_output_dt_pca['RMSE'] = RMSE_array_dt_pca
error_output_dt_pca['MAE'] = MAE_array_dt_pca
error_output_dt_pca


# In[131]:


fig, axs = plt.subplots(1,2, figsize=(15,6))
fig.subplots_adjust(hspace=0.4)
axs=axs.ravel()
N=8
x=np.arange(1,N)
i=0
for col in ['RMSE','MAE']:
    axs[i].bar(x,height=error_output_rdm_pca[col],label = 'Random Forest',width=0.2)
    axs[i].bar(x+0.2,height=error_output_dt_pca[col],label = 'Decision Tree',width=0.2)
    axs[i].legend()
    axs[i].set_title(col+' Error')
    i+=1


# In[132]:


print('Average RMSE Decision Tree Error: {}'.format(error_output_dt_pca.RMSE.mean()))
print('Average RMSE Random Forest Error: {}'.format(error_output_rdm_pca.RMSE.mean()))


# #### 3. Compare the performance of Linear Model and Non-Linear Model from the previous observations. Which performs better and why?
# 

# In[133]:


fig, axs = plt.subplots(1,2, figsize=(15,6))
fig.subplots_adjust(hspace=0.4)
axs=axs.ravel()
N=8
x=np.arange(1,N)
i=0
for col in ['RMSE','MAE']:
    axs[i].bar(x,height=error_output_rdm[col],label = 'Random Forest',width=0.2,color = '#ffa31a')
    axs[i].bar(x+0.2,height=error_output_lrc[col],label = 'Linear Regression',width=0.2,color = '#008ae6')
    axs[i].legend()
    axs[i].set_title(col+' Error')
    i+=1


# In[134]:


print('Average RMSE Random Forest Error: {}'.format(error_output_rdm.RMSE.mean()))
print('Average RMSE Linear Regression Error: {}'.format(error_output_lrc.RMSE.mean()))


# * From the above graph, it is clear that non-linear model i.e. random forest performs better than Linear Regression model.

# ### 4. Train a Time-series model on the data taking time as the only feature. This will be a store-level training.

# ##### a) Identify yearly trends and seasonal months

# * Time Series Analysis

# In[135]:


from statsmodels.tsa.stattools import adfuller,acf,pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA


# In[136]:


def test_stationarity(timeseries):
    rolmean = timeseries.rolling(window=52,center = False).mean()
    rolstd = timeseries.rolling(window = 52,center = False).std()
    plt.figure(figsize=(16,8))
    orig = plt.plot(timeseries,color = '#3399ff',label = 'Original')
    mean = plt.plot(rolmean,color = 'red',label = 'Rolling Mean')
    std = plt.plot(rolstd,color = 'green',label = 'Rolling Std')
    plt.title('Rolling mean and Standard deviation')
    plt.legend(loc='best')
    plt.show(block=False)
 
    print('Result of Dickey-Fuller Test: ')
    dftest = adfuller(timeseries,autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],index=['Test Statistic','p-value','Number of lag used','Number of observation used'])
 
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# In[137]:


Retail_df.head()


# In[138]:


Retail_df.sort_values('Date',inplace=True)


# In[139]:


Retail_df = Retail_df[Retail_df.Open==1]
Retail_df.reset_index(drop = True,inplace=True)


# In[140]:


data = Retail_df[Retail_df.Store==1][['Date','Sales']]
data.set_index('Date',inplace=True)


# In[141]:


data


# In[142]:


test_stationarity(data)


# * p-value is very close to zero so we will reject the null hypothesis, that data does not have a unit root and is stationary.
# 
# * However, data shows some seasonal effects.

# In[143]:


ts_log = np.log(data)
movingavg = ts_log.rolling(window = 12).mean()
plt.figure(figsize=(16,8))
plt.plot(ts_log,color='y',label = 'Log of timeseries data')
plt.plot(movingavg,color='#ff3300',label = 'Moving Average')
plt.legend()
plt.show()


# * From the above graph we can see seasonal effect in the dataset.
# 
# * In dec to jan month sale is high in comaprison to other month

# In[144]:


ts_log_mv_diff = ts_log - movingavg
ts_log_mv_diff.dropna(inplace=True)


# * Since p-value is less than 0.05, so we can say that data is stationary.
# 
# * hence differencing is not required, therefore d = 0
# 

# In[145]:


plt.figure(figsize=(16,8))
plot_pacf(data.dropna(), lags=30)
plt.show()


# * The first lag is the only one vastly above the signicance level and so p = 1.

# In[146]:


plot_acf(data.dropna())
plt.show()


# * Four lag can be found above the significance level and thus q = 4. 

# In[147]:


model = ARIMA(np.array(data[:-6]), order=(1, 0, 4))
results = model.fit()


# In[148]:


plt.figure(figsize=(16,8))
results.plot_predict(700,754)
plt.show()


# In[149]:


results.summary()


# In[150]:


RMSE_ARIMA = np.sqrt(mean_squared_error(np.array(data[700:]) , results.predict(700,753)))
RMSE_ARIMA


# In[151]:


MAE_ARIMA = mean_absolute_error(np.array(data[700:]) , results.predict(700,753))
MAE_ARIMA


# ### Implementing Neural Networks:

# #### 1. Train a LSTM on the same set of features and compare the result with traditional time-series model.

# In[152]:


std = data.std()
mean = data.mean()
timeseries = np.array((data-mean)/std)


# In[153]:


training_size = int(len(timeseries)*0.65)
test_size = len(timeseries)-training_size


# In[154]:


train_size,test_size = timeseries[:training_size,:],timeseries[training_size:len(timeseries),:1]


# In[155]:


def create_dataset(dataset,time_step = 1):
   dataX,dataY = [],[]
   for i in range(len(dataset)-time_step-1):
       a = dataset[i:(i+time_step),0]
       dataX.append(a)
       dataY.append(dataset[i+time_step,0])
   return np.array(dataX),np.array(dataY)


# In[156]:


time_step =100
x_train,y_train = create_dataset(train_size,time_step)
x_test,y_test = create_dataset(test_size,time_step)


# In[157]:


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)


# In[158]:


from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import layers
from keras.optimizers import Adam,RMSprop,SGD,Adagrad
from keras.models import load_model
from sklearn.cluster import KMeans


# In[159]:


model = Sequential()
model.add(LSTM(50,return_sequences = True,input_shape = (100,1)))
model.add(LSTM(50,return_sequences = True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer = 'adam', metrics = ['mae'])
from livelossplot import PlotLossesKerasTF


# In[160]:


model.summary()


# In[161]:


result = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=200,batch_size=20,callbacks = [PlotLossesKerasTF()])


# In[162]:


train_predict = model.predict(x_train)
test_predict = model.predict(x_test)


# In[163]:


#train_predict = std.inverse_transform(train_predict)
train_predict = train_predict.reshape(len(train_predict))
#test_predict = std.inverse_transform(test_predict)
test_predict = test_predict.reshape(len(test_predict))


# In[164]:


# inversion of normalisation
train_predict = train_predict*std.values + mean.values
test_predict = test_predict*std.values + mean.values
y_train = y_train*std.values + mean.values
y_test = y_test*std.values + mean.values


# In[165]:


RMSE_LSTM = np.sqrt(mean_squared_error(y_train,train_predict))
RMSE_LSTM


# In[166]:


RMSE_LSTM = np.sqrt(mean_squared_error(y_test,test_predict))
RMSE_LSTM


# In[167]:


plt.figure(figsize = (16,8))
plt.plot(y_train, label = 'y_test')
plt.plot(train_predict,label = 'y_pred')
plt.title('LSTM')
plt.legend()
plt.show()


# ### 2. Comment on the behavior of all the models you have built so far

# In[168]:


models_error = [[error_lr.RMSE.mean(), 'linear regression model']] # linear regression model
models_error.append([error_output_ada.RMSE.mean(),'Ada-Boost Regression']) # Ada-Boost Regression
models_error.append([error_output_rdg.RMSE.mean(),'Ridge Regression']) # Ridge Regression
models_error.append([error_output_rdm.RMSE.mean(),'Random Forest Regression']) # Random Forest Regression
models_error.append([error_output_lrc.RMSE.mean(),'Linear Regression when store is open']) # Linear Regression when store is open
models_error.append([error_output_dt_pca.RMSE.mean(),'Decision Tree with PCA'])
models_error.append([error_output_rdm_pca.RMSE.mean(),'Random Forest with PCA'])
models_error.append([RMSE_ARIMA,'ARIMA model for a Store'])
models_error.append([RMSE_LSTM, 'LSTM model for a Store'])


# In[169]:


models_error = pd.DataFrame(models_error)


# In[170]:


plt.figure(figsize=(12,6))
plt.barh(models_error[1],models_error[0], color = "r")
plt.title('RMSE Error graph of different models')
plt.show()


# * From the above graph we can clearly says that Random forest performs best out of all models.
# 

# #### 3. Cluster stores using sales and customer visits as features. Find out how many clusters or groups are possible. Also visualize the results.
# 

# In[171]:


cluster_data = new_df[new_df.Open==1]
cluster_data = cluster_data[['Store','Sales','Customers']]
cluster_data.head()


# In[172]:


plt.figure(figsize=(16,8))
sns.scatterplot(data=cluster_data,x='Sales',y='Customers', hue = 'Store', palette = 'magma')
plt.title('Scatter Plot of different Stores')
plt.show()


# In[173]:


kmeans = KMeans(n_clusters=5, random_state=24).fit(np.array(cluster_data[['Sales','Customers']]))


# In[174]:


cluster_data['forecast'] = kmeans.predict(np.array(cluster_data[['Sales','Customers']]))


# In[175]:


cluster_data.head()


# In[176]:


kmeans.labels_


# In[177]:


kmeans.cluster_centers_


# In[178]:


plt.figure(figsize=(16,8))
sns.scatterplot(data=cluster_data,x='Sales',y='Customers', hue = 'forecast')
plt.scatter(
 kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
 s=250, marker='*',
 c='red', edgecolor='black',
 label='centroids'
)
plt.show()


# #### 4. Is it possible to have separate prediction models for each cluster? Compare results with the previous models.
# 

# * We will choose Random Forest Regression and prepare a separate prediction model for each cluster.

# In[179]:


cluster_data = new_df[new_df.Open == 1]
cluster_data.drop('Open',axis=1,inplace=True)


# In[180]:


kmeans = KMeans(n_clusters=5, random_state=24).fit(np.array(cluster_data[['Sales','Customers']]))
cluster_data['forecast'] = kmeans.predict(np.array(cluster_data[['Sales','Customers']]))


# In[181]:


cluster_data.head()


# In[182]:


cluster_data.drop('Store',axis=1,inplace=True)


# In[183]:


RMSE_cluster_rdm = []
MAE_cluster_rdm=[]
cluster = [0,1,2,3,4]
for clust in range(5):
    data = cluster_data[cluster_data.forecast==clust]
    data.drop('forecast',axis=1,inplace=True)
    y=np.array(data['Sales'])
    x=np.array(data.drop('Sales',axis=1))
    y_test,y_pred = random_forest(np.array(x),np.array(y))
    RMSE_1,MAE_1 = error_cal(y_test,y_pred)
    RMSE_cluster_rdm.append(RMSE_1)
    MAE_cluster_rdm.append(MAE_1)


# In[184]:


error_output_cluster_rdm = pd.DataFrame()
error_output_cluster_rdm['cluster'] = cluster
error_output_cluster_rdm['RMSE'] = RMSE_cluster_rdm
error_output_cluster_rdm['MAE'] = MAE_cluster_rdm
error_output_cluster_rdm


# In[185]:


plt.figure(figsize=(10,8))
plt.bar(x=error_output_cluster_rdm.cluster,height=error_output_cluster_rdm.RMSE,label = 'RMSE',width=0.4, color = 'r')
plt.bar(x=error_output_cluster_rdm.cluster,height=error_output_cluster_rdm.MAE,label = 'MAE',width=0.4, color = 'y')
plt.xlabel('Cluster')
plt.legend()
plt.show()


# * since data is not suitable for clustring, we can not separate data into different clusters.
# * so while predicting sales based on clusters, it shows unpredictible result (RMSE, and MAE)

# ### 1. Use ANN (Artificial Neural Network) to predict Store Sales.
# 

# In[186]:


new_df


# In[187]:


new_df = new_df[new_df.Store<=100]
new_df = new_df[new_df.Open == 1]
new_df.reset_index(drop=True, inplace=True)
y = new_df['Sales']
x = new_df.drop(['Sales','Open'],axis=1)
std = StandardScaler()
x = std.fit_transform(x)


# In[188]:


x_train,x_test,y_train,y_test = train_test_split(np.array(x),np.array(y),random_state=42,test_size=0.2)


# In[189]:


model_1 = Sequential()
model_1.add(layers.Dense(32, activation='elu', input_shape = (x_train.shape[1],)))
model_1.add(layers.Dense(64, activation='elu'))
model_1.add(layers.Dense(64, activation='elu'))
model_1.add(layers.BatchNormalization())
## block 2
model_1.add(layers.Dense(128, activation='elu'))
model_1.add(layers.Dense(128, activation='elu'))
model_1.add(layers.BatchNormalization())
## block 3
model_1.add(layers.Dense(256, activation='elu'))
model_1.add(layers.Dense(256, activation='elu'))
model_1.add(layers.BatchNormalization())
## block 4
model_1.add(layers.Dense(128, activation='elu'))
model_1.add(layers.Dense(128, activation='elu'))
model_1.add(layers.BatchNormalization())
## block 5
model_1.add(layers.Dense(64, activation='elu'))
model_1.add(layers.Dense(64, activation='elu'))
model_1.add(layers.Dense(32, activation='elu'))
model_1.add(layers.Dense(1))


# In[190]:


model_1.compile(loss='mse',
 optimizer = Adam(learning_rate=0.001),
 metrics=['mae'])


# In[191]:


result1 = model_1.fit(x_train,y_train,epochs=100,batch_size=20,verbose=1,callbacks = [PlotLossesKerasTF()])


# In[192]:


y_pred = model_1.predict(x_test)


# In[193]:


RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
RMSE


# ##### Import test data for predict

# In[194]:


from sklearn.preprocessing import LabelEncoder


# In[195]:


test_data = hidden_data
test_data


# In[196]:


test_data.drop('Date',axis = 1, inplace=True)
test_data.loc[test_data.StateHoliday==0,'StateHoliday'] = '0'
labelencoder= LabelEncoder()
test_data.StateHoliday = labelencoder.fit_transform(test_data['StateHoliday'])
test_data = test_data[test_data.Store<=100]
test_data = test_data[test_data.Open == 1]
test_data.reset_index(drop=True, inplace=True)
y = test_data['Sales']
x = test_data.drop(['Sales','Open'],axis=1)
std = StandardScaler()
x = std.fit_transform(x)


# In[197]:


y_pred = model_1.predict(x)
np.sqrt(mean_squared_error(y,y_pred))


# In[198]:


plt.figure(figsize=(16,8))
plt.plot(y_pred[:100],label = 'sales forecast', color = 'b')
plt.plot(y[:100],label = 'Actual sales', color = 'g')
plt.legend()
plt.title('Actual vs Forecasted Sales\nModel trained on 100 Stores')


# #### 2. Use Dropout for ANN and find the optimum number of clusters (clusters formed considering the features: sales and customer visits). Compare model performance with traditional ML based prediction models.
# 

# In[199]:


new_df = new_df[new_df.Store<=100]
new_df = new_df[new_df.Open == 1]
new_df.reset_index(drop=True, inplace=True)
y = new_df['Sales']
x = new_df.drop(['Sales','Open'],axis=1)
std = StandardScaler()
x = std.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(np.array(x),np.array(y),random_state=42,test_size=0.2)


# In[200]:


model_2 = Sequential()
model_2.add(layers.Dense(32, activation='relu', input_shape = (x_train.shape[1],)))
model_2.add(layers.Dense(64, activation='relu'))
model_2.add(layers.Dense(64, activation='relu'))
model_2.add(layers.BatchNormalization())
## block 2
model_2.add(layers.Dense(128, activation='relu'))
model_2.add(layers.Dense(128, activation='relu'))
model_2.add(layers.BatchNormalization())
## block 3
model_2.add(layers.Dense(256, activation='relu'))
model_2.add(layers.Dense(256, activation='relu'))
model_2.add(layers.BatchNormalization())
model_2.add(layers.Dropout(0.5))
## block 4
model_2.add(layers.Dense(128, activation='relu'))
model_2.add(layers.Dense(128, activation='relu'))
model_2.add(layers.BatchNormalization())
model_2.add(layers.Dropout(0.5))
## block 5
model_2.add(layers.Dense(64, activation='relu'))
model_2.add(layers.Dense(64, activation='relu'))
model_2.add(layers.Dense(32, activation='relu'))
model_2.add(layers.Dropout(0.5))
model_2.add(layers.Dense(1, activation = 'relu'))


# In[201]:


model_2.compile(loss='mse',
 optimizer = Adam(learning_rate=0.001),
 metrics=['mae'])


# In[202]:


model_2.summary()


# In[203]:


result2 = model_2.fit(x_train,y_train,epochs=200,batch_size=20)


# In[204]:


y_pred = model_2.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))


# ### 3. Find the best setting of neural net that minimizes the loss and can predict the sales best. Use techniques like Grid search,cross-validation and Random search.

# In[207]:


def modelkf(x_train,y_train,x_test,y_test):
    model = Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape = (x_train.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
     ## block 2
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    ## block 3
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    ## block 4
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    ## block 5
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation = 'relu'))
 
    model.compile(loss='mse',
    optimizer = Adam(learning_rate=0.01),
      metrics=['mae'])
 
    model.fit(x_train,y_train,epochs=70,batch_size=50)
 
    y_pred = model.predict(x_test)
 
    return (np.sqrt(mean_squared_error(y_test,y_pred)))


# In[208]:


score_kf_ann = []
kf = StratifiedKFold(n_splits=4)
for train_index,test_index in kf.split(x,y):
    x_train,x_test,y_train,y_test = train_test_split(np.array(x),np.array(y),test_size = 0.20,random_state=42)
    score_kf_ann.append(modelkf(x_train,y_train,x_test,y_test))


# In[210]:


score_kf_ann


# In[211]:


RMSE = sum(score_kf_ann)/len(score_kf_ann)
RMSE


# In[ ]:




