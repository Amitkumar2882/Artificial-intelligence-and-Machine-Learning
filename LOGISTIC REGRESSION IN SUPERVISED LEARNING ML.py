#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


titanic_df = pd.read_csv('https://raw.githubusercontent.com/ammishra08/MachineLearning/master/Datasets/Titanic_Data.csv') 
titanic_df.head()


# #### Data Wrangling 

# In[3]:


titanic_df.isnull().sum()


# #### EDA - Exploratory Data Analysis

# In[4]:


plt.figure(figsize = (8,6))
sns.countplot(x = 'Pclass',data = titanic_df, palette = 'plasma')


# In[5]:


titanic_df['Pclass'].value_counts()


# * Distribution of Passengers in all three class. Most passengers are travelling in Pclass = 3 in Titanic Ship

# In[6]:


plt.figure(figsize = (7,7))
sns.countplot(x = 'Survived', data = titanic_df, palette = 'Set1')


#    * Distribution of Survived and Not Survived. Here 61.62% of Total Passengers Not Survived in Titanic Ship

# In[7]:


titanic_df['Survived'].value_counts()/len(titanic_df)*100


# In[8]:


# hue = category (Pclass) 
sns.catplot(x = 'Survived', kind = 'count', hue = 'Pclass', data = titanic_df, palette = 'plasma')


# In[9]:


# hue = category (Pclass) 
sns.catplot(x = 'Pclass', kind = 'count', hue = 'Survived', data = titanic_df, palette = 'Set1')


# In[10]:


# cross tab
contigency_table = pd.crosstab(titanic_df['Pclass'], titanic_df['Survived'])
contigency_table


# In[11]:


# hue = category (Pclass) 
# catplot is categorical plot,kind of plot(satter plot, bar plot etc)
sns.catplot(x = 'Survived', kind = 'count', hue = 'Sex', data = titanic_df, palette = 'Set1')


# #### Handling the missing values

# In[12]:


sns.set_style('whitegrid')
plt.figure(figsize = (12,8))
sns.boxplot( x = 'Pclass', y = 'Age', data = titanic_df, palette = 'plasma')


# In[13]:


# Alternative approach to replace Missing Age by Title of Name
titanic_df['Title'] = titanic_df['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)


# In[14]:


titanic_df['Title'].unique()


# In[15]:


pd.crosstab(titanic_df['Title'],titanic_df['Sex'])


# In[16]:


titanic_df['Title'] = titanic_df['Title'].replace(['Don','Rev','Dr','Major','Lady','Sir','Col','Capt','Countess','Jonkheer'], 'Rare')


# In[17]:


titanic_df['Title'] = titanic_df['Title'].replace('Mlle','Miss')
titanic_df['Title'] = titanic_df['Title'].replace('Ms','Miss')
titanic_df['Title'] = titanic_df['Title'].replace('Mme','Mrs')


# In[18]:


titanic_df[['Title','Age']].groupby(['Title']).median()


# #### UDF - Simple Imputer
#     * Replacing Missing Values with median

# In[19]:


def imputer_age(cols):
    Age = cols[0]
    Title = cols[1]
    if pd.isnull(Age):
        if Title == 'Master':
            return 3.5
        elif Title == 'Miss':
            return 21
        elif Title == 'Mr':
            return 30
        elif Title == 'Mrs':
            return 35
        else:
            return 48.5
    else:
        return Age
            
            
        


# In[20]:


titanic_df['Age'] = titanic_df[['Age','Title']].apply(imputer_age, axis = 1)


# In[21]:


titanic_df.isnull().sum()


# In[22]:


titanic_df.drop('Cabin', axis = 1, inplace = True)


# In[23]:


titanic_df.dropna(inplace = True)


# #### Data Preprocessing

# In[24]:


titanic_df.head()


# In[25]:


# one Hot Encoding
sex = pd.get_dummies(titanic_df['Sex'],drop_first = True)
sex


# In[26]:


# one Hot Encoding
title = pd.get_dummies(titanic_df['Title'],drop_first = True)
title


# In[27]:


embarked = pd.get_dummies(titanic_df['Embarked'],drop_first = True)
embarked


# In[28]:


titanic_df1 = pd.concat([titanic_df,sex,title,embarked], axis = 1)
titanic_df1


# #### Feature & Target for Model

# In[29]:


X_feature = titanic_df1.drop(['PassengerId','Survived','Name','Sex','Ticket','Title','Embarked'], axis = 1)
Y_target = titanic_df1['Survived']


# ####  Randam Sampling
#     * Splitting Data into Train & Test

# In[30]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_feature ,Y_target, test_size = 0.2, random_state = 1)


# ####  Logistic Regression

# In[31]:


from sklearn.linear_model import LogisticRegression
#vsolver = 'lbfgs', c = higher the better , max_iter = 1e7
logit_model = LogisticRegression(solver = 'lbfgs', C = 1e5, max_iter = 1e7,penalty = 'l2')


# In[32]:


logit_model.fit(x_train , y_train)


# In[33]:


# Accuracy
logit_model.score(x_test , y_test)


# #### Classification Matrics
#     * Confusion Matrix
#     * Classification Report

# In[34]:


predictions = logit_model.predict(x_test)
predictions


# In[35]:


from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_test , predictions)


# In[36]:


# Diagonals are True Predictions and Non-Diagonals are False
sns.heatmap(confusion_matrix(y_test, predictions), annot = True ,fmt = '0.0f')


# In[37]:


print(classification_report(y_test, predictions))


# #### Make New Prediction

# In[38]:


X_feature


# In[68]:


X_feature['Age'].sort_values(ascending = False)


# In[78]:


x_jack = [[3,22.0,0,0,8.55,1,0,1,0,0,1,0]]
x_rose = [[1,80,1,0,77,0,1,0,1,1,1,0]]


# In[79]:


logit_model.predict(x_jack)


# In[80]:


logit_model.predict(x_rose)


# In[ ]:




