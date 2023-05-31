# Artificial-intelligence-and-Machine-Learning
#!/usr/bin/env python
# coding: utf-8

# ## Retail - RFM (Recency Frequency Monetary) model

# ### DESCRIPTION

# * It is a critical requirement for business to understand the value derived from a customer. RFM is a method used for analyzing customer value.
# 
# * Customer segmentation is the practice of segregating the customer base into groups of individuals based on some common characteristics such as age, gender, interests, and spending habits

# ### OBJECTIVE

# * Perform customer segmentation using RFM analysis.
# 
# *  k-means clustering algorithm Model

# ### Project Task: Week 1

# #### import important libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from operator import attrgetter
import matplotlib.colors as mcolors
import datetime as dt
from scipy.stats import skewnorm
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
import pylab as p
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_excel("Online Retail.xlsx" , sheet_name = "Online Retail")


# In[3]:


data


# ### Data Cleaning 

# * Check missing data

# In[4]:


data.isnull().sum()


# ##### Drop missing data 

# In[5]:


data.dropna(subset = ['CustomerID'] , inplace = True)


# In[6]:


data.isnull().sum()


# ##### Remove Duplicate Data from dataset

# In[7]:


data.duplicated().sum()


# In[8]:


data = data.drop_duplicates()


# In[9]:


data.duplicated().sum()


# In[10]:


data.shape


# #### Exploratory Data Analysis

# In[11]:


pd.DataFrame(data['Country'].unique())


# ##### Total numbers of customers

# In[12]:


len(data['CustomerID'].unique())


# #### Numbers of customers countrywise

# In[13]:


customers = pd.DataFrame(data.groupby('Country')['CustomerID'].nunique())


# In[14]:


customers_countrywise = pd.DataFrame(customers).sort_values(by = 'CustomerID', ascending = False)
customers_countrywise


# ##### Customer order more than one item

# In[15]:


n_order = data.groupby(['CustomerID'])['InvoiceNo'].nunique()
multiple_order_percentage = np.sum(n_order>1) / data['CustomerID'].nunique()
print(f'{100*multiple_order_percentage:.2f}%')


# ax = sns.distplot(n_order, kde = False , hist = True)
# ax.set(title = "Distribution of number of orders per customers",
#       xlabel = 'no. of orders',
#       ylabel = 'no. of customers');
# 

# ### Data Transformation

# ##### Cohort Analysis of dataset

# In[16]:


def get_month(x):
    return dt.datetime(x.year, x.month, 1)


# In[17]:


data['InvoiceMonth'] = data['InvoiceDate'].apply(get_month)


# In[18]:


data['CohortMonth'] = data.groupby('CustomerID')['InvoiceMonth'].transform('min')


# In[19]:


# Creating Cohort Idex to track the month lapse between that specific transaction and the first transaction that user made. 
def get_date(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    day = df[column].dt.day
    return year, month, day


# In[20]:


invoice_year, invoice_month, _ = get_date(data, 'InvoiceMonth')
cohort_year, cohort_month, _ = get_date(data, 'CohortMonth')


# In[21]:


year_diff = invoice_year - cohort_year
month_diff = invoice_month - cohort_month


# In[22]:


data['CohortIndex'] = year_diff * 12 + month_diff + 1


# ##### Active customer each cohort

# In[23]:


cols=['CohortMonth','CohortIndex']


# In[24]:


cohort_data = data.groupby(cols)['CustomerID'].apply(pd.Series.nunique).reset_index()


# In[25]:


cohort_data.rename(columns={'CustomerID':'No_of_Custs'},inplace=True)


# In[26]:


cohort_data.head()


# In[27]:


cohort_count = cohort_data.pivot_table(index='CohortMonth',columns='CohortIndex',values='No_of_Custs')


# In[28]:


cohort_count


# In[29]:


# month list:
month_list = ["Dec '10", "Jan '11", "Feb '11", "Mar '11", "Apr '11",              "May '11", "Jun '11", "Jul '11", "Aug '11", "Sep '11",               "Oct '11", "Nov '11", "Dec '11"]


# In[30]:


# to represent the same as percentage is below:
cohort_size = cohort_count.iloc[:,0]
#cohort_size
retention = cohort_count.divide(cohort_size, axis = 0)
#retention
retention.round(3) * 100


# ##### Visual representation 

# In[31]:


plt.figure(figsize = (16,7))
plt.title('Cohort Analysis - Retention Rate')
sns.heatmap(data = retention, 
            annot = True,
            cmap = "Set1",
            vmin = 0.0,
            vmax = 0.6,
            fmt = '.1%', 
            linewidth = 0.3,
            yticklabels=month_list
           )
plt.show()


# In[ ]:





# ### Project Task: week 2

# ##### Build RFM Model

# In[32]:


# creating a new column TotalPrice= quantity * UnitPrice
data['TotalPrice'] = data['Quantity']*data['UnitPrice']


# In[33]:


data['MonetaryValue']= data['TotalPrice']


# In[34]:


data.head(10)


# In[35]:


# Now to calculate TotalPrice spend by each customer: (Monetary)
df_monetary = data.groupby(['CustomerID']).agg({'MonetaryValue':sum}).reset_index()
df_monetary.head(5)


# In[36]:


# Now to calculate how frequently the cutomer made the purchase: (Frequency)
df_freq = data.groupby(['CustomerID'])['CohortIndex'].nunique().reset_index()

# renaming the column name to Frequency 
df_freq.rename(columns={'CohortIndex':'Frequency'},inplace=True)
df_freq.head(10)


# In[37]:


# Now to Calculate Recency: How recent the customer purchased, can be calculated by getting the diff in first and last 
# transaction dates.
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'],format='%d-%m-%Y %H:%M')
max_date= max(data['InvoiceDate'])

#difference between last date and transaction date
data['Days_diff'] = max_date-data['InvoiceDate']

#To get only Days in 'Days_diff'
data['Days_diff'] = data['Days_diff'].dt.days


# In[38]:


data.head(10)


# In[39]:


# recency per customer (last transaction date)
df_recency = data.groupby('CustomerID')['Days_diff'].min().reset_index()
df_recency.head(10)


# In[40]:


# Now merge R, F and M dataframe to get RFM Dataframe

df_rf = pd.merge(df_recency,df_freq,on='CustomerID',how='inner')
df_rfm = pd.merge(df_rf,df_monetary,on='CustomerID',how='inner')


# In[41]:


df_rfm.columns=['CustomerID','Recency','Frequency','MonetaryValue']
df_rfm.head(10)


# In[42]:


# Create labels and groups, then assign them to three percentile groups
#r_labels = range(4, 0, -1)
r_labels = [4,3,2,1]
r_groups = pd.qcut(df_rfm.Recency,q=4,labels=r_labels)


#f_labels = range(1, 5)
f_labels = [1,2,3,4]
f_groups = pd.qcut(df_rfm.Frequency.rank(method="first"),q=4,labels=f_labels) 
#f_groups = pd.qcut(df_rfm.Frequency,q=4,labels=f_labels) 

#m_labels = range(1, 5)
m_labels = [1,2,3,4]
m_groups = pd.qcut(df_rfm.MonetaryValue,q=4,labels=m_labels) 


# In[43]:


# Create new columns in RFM dataframe with R,F,M columns
df_rfm['R'] = r_groups.values
df_rfm['F'] = f_groups.values
df_rfm['M'] = m_groups.values


# In[44]:


df_rfm


# In[45]:


# combining these 3 columns(R,F,M) to get the RFM segment and RFM score
df_rfm['RFM_Segment'] = df_rfm.apply(lambda x: str(x['R']) + str(x['F']) + str(x['M']),axis=1)
df_rfm['RFM_Score'] = df_rfm[['R','F','M']].sum(axis=1)


# In[46]:


df_rfm


# In[47]:


#let's do segmentation
segment_dict = {    
    'Best Customers':'444',      # Highest frequency as well as monetary value with least recency
    'Loyal Customers':'344',     # High frequency as well as monetary value with good recency
    'Potential Loyalists':'434', # High recency and monetary value, average frequency
    'Big Spenders':'334',        # High monetary value but good recency and frequency values
    'At Risk Customers':'244',   # Customer's shopping less often now who used to shop a lot
    'Can’t Lose Them':'144',      # Customer's shopped long ago who used to shop a lot.
    'Recent Customers':'443',    # Customer's who recently started shopping a lot but with less monetary value
    'Lost Cheap Customers':'122' # Customer's shopped long ago but with less frequency and monetary value
}


# In[48]:


# Swap the key and value of dictionary
dict_segment = dict(zip(segment_dict.values(),segment_dict.keys()))


# In[49]:


df_rfm['Segment'] = df_rfm['RFM_Segment'].map(lambda x: dict_segment.get(x))


# In[50]:


# Fill other scores with 'Others'
df_rfm.Segment.fillna('Others', inplace=True)


# In[51]:


df_rfm


# In[52]:


RFM_Segment_Counts = df_rfm[df_rfm.Segment!='None'].groupby('Segment')['CustomerID'].count().reset_index(name='counts')


# In[53]:


RFM_Segment_Counts


# #### Data Modeling

# In[54]:


# import required libraries for clustering
from scipy.stats import norm
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[55]:


f,ax = plt.subplots(1,3, figsize=(23,12))
sns.despine(left=True)
x = pd.Series(df_rfm['Recency'], name="Recency")
sns.distplot(x,fit=norm, kde=False,ax=ax[0])
y = pd.Series(df_rfm['Frequency'], name="Frequency")
sns.distplot(y,fit=norm, kde=False,ax=ax[1])
z = pd.Series(df_rfm['MonetaryValue'], name="MonetaryValue")
sns.distplot(z,fit=norm, kde=False,ax=ax[2])
plt.show()


# In[56]:


# From the third plot(Monetary Value), extreme skewness can be seen. Need to be treated. Lets plot Boxplot for each
# set up the axes of subplot:
f,ax = plt.subplots(1,3, figsize=(23,12))
sns.despine(left=True)
cols=['Recency','MonetaryValue','Frequency']
sns.boxplot(cols[0],data=df_rfm,orient='v',ax=ax[0],width=0.3)
sns.boxplot(cols[1],data=df_rfm,orient='v',ax=ax[1],width=0.3)
sns.boxplot(cols[2],data=df_rfm,orient='v',ax=ax[2],width=0.3)
plt.show()


# * From both the above plot representation it can be seen that the Monetary_value is having extreme skewness. Need to be treated.

# In[57]:


# Treating Outliers for MonetaryValue:
Q1 = df_rfm.MonetaryValue.quantile(0.05)
Q3 = df_rfm.MonetaryValue.quantile(0.95)

IQR = Q3 - Q1
df_rfm = df_rfm[(df_rfm.MonetaryValue>=Q1 - 1.5*IQR)&(df_rfm.MonetaryValue<=Q3 + 1.5*IQR)]


# In[58]:


# Treating Outliers for Frequency:
Q1 = df_rfm.Frequency.quantile(0.05)
Q3 = df_rfm.Frequency.quantile(0.95)

IQR = Q3 - Q1
df_rfm = df_rfm[(df_rfm.Frequency>=Q1 - 1.5*IQR)&(df_rfm.Frequency<=Q3 + 1.5*IQR)]


# In[59]:


# Treating Outliers for Recency:
Q1 = df_rfm.Recency.quantile(0.05)
Q3 = df_rfm.Recency.quantile(0.95)

IQR = Q3 - Q1
df_rfm = df_rfm[(df_rfm.Recency>=Q1 - 1.5*IQR)&(df_rfm.Recency<=Q3 + 1.5*IQR)]


# #### Rescaling the Attributes using standardization

# In[60]:


rfm_df = df_rfm[['Recency','Frequency','MonetaryValue']]

# Instantiate
scaler = StandardScaler()

# fit_transform
rfm_df_scaled = scaler.fit_transform(rfm_df)
rfm_df_scaled.shape


# In[61]:


rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
rfm_df_scaled.columns = ['Recency','Frequency','MonetaryValue']
rfm_df_scaled.head()


# #### KNN Model

# In[62]:


kmeans = KMeans(n_clusters=4, max_iter=50)
kmeans.fit(rfm_df_scaled)


# In[63]:


kmeans.labels_


# In[64]:


# Elbow-curve/SSD

ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    
    ssd.append(kmeans.inertia_)
    
# plot the SSDs for each n_clusters
plt.plot(ssd)


# In[65]:


# Silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(rfm_df_scaled, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))


# In[66]:


# Final model with k=3
kmeans = KMeans(n_clusters=3, max_iter=50)
kmeans.fit(rfm_df_scaled)


# In[67]:


kmeans.labels_


# In[68]:


# assign the label
df_rfm['Cluster_Id'] = kmeans.labels_
df_rfm.head()


# ##### The optimum number of clusters to be formed = 3.

# In[69]:


f,ax = plt.subplots(1,3,figsize=(15,9))
sns.boxplot(x='Cluster_Id', y='MonetaryValue', data=df_rfm,ax=ax[0])
sns.boxplot(x='Cluster_Id', y='Recency', data=df_rfm,ax=ax[1])
sns.boxplot(x='Cluster_Id', y='Frequency', data=df_rfm,ax=ax[2])
plt.show()


# ### Inference:

# ##### 1. CustomerIDs with clusterID as 1 are the customers with high amount of transactions as compared to other customers.

# ##### 2. But, customers with clusterID 1 are not recent buyers and hence can't lose the customers with clusterID 1. Some Value added services should be given to them in business point of view.

# ##### 3. Customers with clusterID 0 are recent buyers but there transaction amount is lesser than other two clusterIDs and hence not much attention is required on them.

# In[ ]:



