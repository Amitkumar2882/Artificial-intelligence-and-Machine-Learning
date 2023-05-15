#!/usr/bin/env python
# coding: utf-8

# ## To Identify "Hate Speech" (Racist or Sexist tweets) in "Twitter" using NLP and Machine Learning

# #### Problem Statement

# * Twitter is the biggest platform where anybody and everybody can have their views heard. Some of these voices spread hate and negativity. Twitter is wary of its platform being used as a medium  to spread hate. 

# * You are a data scientist at Twitter, and you will help Twitter in identifying the tweets with hate speech and removing them from the platform. You will use NLP techniques, perform specific cleanup for tweets data, and make a robust model.

# #### Analysis to be done

# * Clean up tweets and build a classification model by using NLP techniques, cleanup specific for tweets data, regularization and hyperparameter tuning using stratified k-fold and cross validation to get the best model.

# * Content: 
#     
#     * id: identifier number of the tweet
#     
#     * Label: 0 (non-hate) /1 (hate)
# 
#     * Tweet: the text in the tweet

# #### 1. import important libraries 

# * pandas for Data Analysis.
# 
# * NumPy for Numerical Operations.
# 
# * Matplotlib/seaborn for Plotting or Data Visualization.
# 
# * Scikit-Learn for Machine Learning Modelling and Evaluation.
# 
# * Text summarization is a subdomain of Natural Language Processing (NLP) that deals with extracting summaries from huge chunks of texts

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import warnings
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)


# #### 2.import dataset and load data

# In[2]:


# import dataset
hate_speech_df = pd.read_csv('TwitterHate.csv')
hate_speech_df


# In[3]:


hate_speech_df.info()


# In[4]:


# Making copy of given dataset
df = hate_speech_df.copy()
df


# #### 3. Exploratory Data Analysis

# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


# Missing values Analysis
df.isnull().sum()


# In[8]:


df.drop('id', inplace = True, axis = 1)


# In[9]:


df


# In[10]:


df['label'].value_counts()


# In[11]:


# Plot the value counts with a bar graph (for visualization) using "Matplotlib"
plt.title('Twitter Hate Speech Analysis')
plt.xlabel('0 = Positive [+ve] Tweets (majority class) |  1 = Negative [-ve] Tweets (minority class)')
plt.ylabel('Frequency Count')
df["label"].value_counts().plot(kind="bar", color=["green","red"])


# * Since two values are not equal data is unbalanced

# ### Text Data Preprocessing

# #### Task 3. To "Clean-up" the Dataset

# *  Normalize the casing
# 
# * Using regular expressions, remove "user handles". These begin with '@’.
# 
# * Using regular expressions, Remove "URLs".
# 
# * Using "TweetTokenizer" from NLTK, Tokenize the tweets ---> Individual terms.
# 
# * Remove "Stop - Words".
# 
# * Remove "Redundant" terms like ‘amp’, ‘rt’, etc.
# 
# * Remove ‘#’ symbols from the tweet while retaining the term.

# In[12]:


import nltk
nltk.download('punkt')
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk import TweetTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords
from autocorrect import Speller
spell = Speller(lang='en')
nltk.download('wordnet')
import string
import re
from wordcloud import WordCloud, STOPWORDS 


# In[13]:


def preprocess_tweet_text(tweet):
    tweet.lower()
    tweet = re.sub('[^A-Za-z0-9]+', ' ', tweet)

    
     # Remove redundant terms like ‘amp’, ‘rt’, etc.
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+ | \#','', tweet)
    
    # Remove urls
    tweet = re.sub(r"http\S+ | www\S+ | https\S+", '', tweet, flags=re.MULTILINE)
    
    # Remove punctuations from tweet - Efficiently using "Translate" method
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    
    #Using TweetTokenizer from NLTK, tokenize the tweets into individual terms.
    tk = TweetTokenizer()
    tweet_tokens = tk.tokenize(tweet)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in tweet_tokens if not w in stop_words]
    
    # Remove redundant terms like ‘amp’, ‘rt’, etc.
    filtered_words_final = [w for w in filtered_words if not w in ('amp', 'rt')]

    return " ".join(filtered_words_final)   
      


# In[14]:


# Apply the pre-processing function for the "tweet" column
df["tweet"] = df["tweet"].apply(preprocess_tweet_text)


# In[15]:


df.head()


# ####  Extra clean-up by removing terms with a length of 1.

# In[16]:


df['length']= df['tweet'].apply(len)


# In[17]:


df.head()


# In[18]:


# Let's check the "Length" of the string and complete the task assigned.

# [a] If "Length" of the string = 0

len(df[df['length'] == 0])
print("The Number of Strings in the tweet column having Length = 0 is {}".format(len(df[df['length'] == 0]))) 

# [b] If "Length" of the string = 1

len(df[df['length'] == 1])
print("The Number of Strings in the tweet column having Length = 1 is {}".format(len(df[df['length'] == 1])))

# [c] If "Length" of the string > 1

len(df[df['length'] > 1])
print("The Number of Strings in the tweet column having Length > 1 is {}".format(len(df[df['length'] > 1])))


# In[19]:


# Now, Remove tweets having length of 1 in our dataframe.

df = df[df["length"] > 1]


# In[20]:


df.shape


# In[21]:


# Function to Apply the "tokenization" process to the text corpus
def token_text(text):
    
    # Tokenize the text into a list of words
    tokens = nltk.tokenize.word_tokenize(text)
    return tokens


# In[22]:


# Final list with "Tokenized words"
tokenized_terms = []

# Iterating over each string in the data
for x in df['tweet']:
    
    # Calling the pre-processed "text" function
    token = token_text(x)
    # Append them
    tokenized_terms.append(token)

# Create "All the tokenized terms" ---> "One Large List"
tokeninized_list = [i for j in tokenized_terms for i in j]


# In[23]:


# Just cross-check the "Type" of "object" created for the text corpus
type(tokeninized_list)


# In[24]:


from collections import Counter


# In[25]:


# Create a "List Comprehension" using "Counter" Class 
most_common_words= [word for word, word_count in Counter(tokeninized_list).most_common(10)]

# Printing the Top 10 most common terms
print(most_common_words)


# In[26]:


# create an "object" for Word Cloud
df_word_cloud = df['tweet'][20000:]

# Plot size in Matplotlib
plt.figure(figsize = (20,20))

# Define the pre-defined stopwords
stopwords = set(STOPWORDS)

# create an object with the name "word_cloud"
word_cloud = WordCloud(max_words = 1000 , width = 1600 , height = 900, collocations=False, stopwords=STOPWORDS).generate(" ".join(df_word_cloud))

# Display the "Word Cloud" as an Image
plt.imshow(word_cloud)


# #### Spilit data into train and test

# In[27]:


from sklearn.model_selection import train_test_split , GridSearchCV , cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit , StratifiedKFold   
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer


# In[28]:


# Split data into "X" and "y"
X = df["tweet"]
y = df["label"]


# In[29]:


# Split into train & test set
X_train, X_test, y_train, y_test = train_test_split(X,  # independent variables
                                                    y,  # dependent variable
                                                    test_size=0.25, random_state = 42)


# ### Transforming Dataset using TF-IDF Vectorizer

# In[30]:


# Instantiate with a maximum of 5000 terms in your vocabulary.
tfidf_vectorizer = TfidfVectorizer(max_features = 7000)


# In[31]:


# Fit and apply on the train set.
X_train = tfidf_vectorizer.fit_transform(X_train)


# In[32]:


# Apply on the test set.
X_test = tfidf_vectorizer.transform(X_test)


# In[33]:


# Let's check the "Shape" of the DataFrame after splitting it

# (rows, columns)
print(f"X_train:{X_train.shape}"),

print(f"X_test:{X_test.shape}")


# #### Logistic Regression Model

# In[34]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score


# In[35]:


model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# #### Model Evaluation: Accuracy, recall, and f_1 score.

# In[36]:


print("Training Score", model.score(X_train, y_train))
print("Testing Score",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# * Recall is low for class 1 i.e 30%

# #### ROC Curve of model at Default Parameter

# In[37]:


# Calculate fpr, tpr and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Calculate roc_auc
roc_auc = auc(fpr, tpr)

# Plot the figure in the notebook
plt.figure(figsize=(12,8))

# Plot ROC curve
plt.plot(fpr, tpr, color='red', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
# Plot line with no predictive power (baseline)
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='No Skill')

# Set the "x-limits" and "y-limits" of the current axes.
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# Customize the plot
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - "Default Parameters"')
plt.legend(loc="lower right")
plt.show()


# In[38]:


# Area under ROC curve
roc_auc_score(y_test, y_pred)
print(f"AUC_GridSearchCV: {roc_auc_score(y_test, y_pred) * 100:.2f}%")


# #### Adjust the im-balance class for better result

# In[39]:


# Instantiate "Logistic Regression" from sklearn --> To find the "class weights"
lr_adjust_class_weights_check = LogisticRegression(random_state = 42, max_iter=10000)

#Setting the range for class weights
weights = np.linspace(0.0,0.99,200)

#Creating a dictionary grid for grid search
param_grid = {'class_weight': [{0:x, 1:1.0-x} for x in weights]}

#Fitting grid search to the train data with 5 folds
gridsearch_adjust = GridSearchCV(estimator= lr_adjust_class_weights_check, 
                                 param_grid= param_grid,
                                 cv=StratifiedKFold(), 
                                 n_jobs=-1,  # Set n_jobs to -1 to use all cores (NOTE: n_jobs=-1 is broken as of 8 Dec 2019, using n_jobs=1 works)
                                 scoring='f1', 
                                 verbose=2).fit(X_train, y_train) # print out progress

#Ploting the score for different values of weight
sns.set_style('whitegrid')
plt.figure(figsize=(15,10))
weight_data = pd.DataFrame({ 'score': gridsearch_adjust.cv_results_['mean_test_score'], 'weight': (1- weights)})
sns.lineplot(weight_data['weight'], weight_data['score'])
plt.xlabel('Weight for class 1 = Negative [-ve] Tweets (minority class)', fontsize = 15)
plt.ylabel('F1 score', fontsize = 15)
plt.xticks([round(i/10,1) for i in range(0,11,1)])
plt.title('Scoring for different class weights', fontsize=25)


# In[40]:


best_accuracy = gridsearch_adjust.best_score_
best_accuracy


# In[41]:


best_parameters = gridsearch_adjust.best_params_
best_parameters


# #### OBSERVATIONS :-

# * Through the graph, we can see that the "Highest value" for the "Minority class" is peaking at about 0.88 class weight.

# * Using Grid search, we got the "Best class weight", (i.e)
# 
#     * 0: 0.11 for class 0 (majority class),
# 
#     * 1: 0.88 for class 1 (minority class).

# In[42]:


# Adjust the "Appropriate Class" in the "Logistic Regression model"
lr_adjust_class_weights = LogisticRegression(class_weight={0: 0.11, 1: 0.88})


# #### Train model again with adjustment in parameter and evaluate

# In[43]:


# "Train" the model on the "train" set (i.e) Fit into  the "train" data
lr_adjust_class_weights.fit(X_train, y_train)


# In[44]:


y_pred_LR = lr_adjust_class_weights.predict(X_test)
print("Training Score", lr_adjust_class_weights.score(X_train, y_train))
print("Testing Score",accuracy_score(y_test, y_pred_LR))


# In[45]:



print(classification_report(y_test, y_pred_LR))


# In[ ]:





# * Recall value for class 1 is 62% which is good in comparison to default parameter model

# #### ROC Curve with Adjusted Parameter of model

# In[46]:


# Evaluate the predictions on the "test" set 
y_test_predict_adjust_class_weights = lr_adjust_class_weights.predict(X_test)

# Calculate fpr, tpr and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_test_predict_adjust_class_weights)

# Calculate roc_auc
roc_auc = auc(fpr, tpr)

# Plot the figure in the notebook
plt.figure(figsize=(12,8))

# Plot ROC curve
plt.plot(fpr, tpr, color='orange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
# Plot line with no predictive power (baseline)
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='No Skill')

# Set the "x-limits" and "y-limits" of the current axes.
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# Customize the plot
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - "Adjusted Class weights Parameters"')
plt.legend(loc="lower right")
plt.show()


# #### Area under the Curve (AUC) --- " Adjusted Class weight Parameters "

# In[47]:


roc_auc_score(y_test, y_test_predict_adjust_class_weights)
print(f"AUC_GridSearchCV: {roc_auc_score(y_test, y_test_predict_adjust_class_weights) * 100:.2f}%")


# ####  "Regularization" and "Hyperparameter tuning" :

# * a) Import "GridSearch" and "StratifiedKFold" because of "Class Im-balance".

# * b) Provide the parameter grid to choose for ‘C’ and ‘penalty’ parameters. 

# * c) Use a Balanced class weight while Instantiating the "Logistic Regression".

# In[48]:


# Provide the parameter grid to choose for ‘C’ and ‘penalty’ parameters.

# Define the "C values" parameters
C_values = np.arange(0.5, 20.0, 0.5)

# Define the "Penalty values" parameters
penalty_values = ["l1", "l2"]

# Define the "Hyperparameter Grid" parameters
hyperparam_grid = {"penalty": penalty_values, "C": C_values }


# In[49]:


# Instantiate "Logistic Regression" from sklearn with "Balanced class weight"

lr_balanced_class_weights = LogisticRegression(solver='liblinear' , class_weight={0: 0.11, 1: 0.88},random_state = 42)


# #### Find the parameters with the "Best Recall" in cross-validation.

# * a) Choose ‘recall’ as the metric for scoring.
# 
# * b) Choose a "Stratified 4 fold cross-validation" scheme.
# 
# * c)  Fit into the "train" set.

# In[50]:


#Fitting grid search to the train data with 5 folds and Fit into the "train" data

gridsearch_balanced = GridSearchCV(estimator= lr_balanced_class_weights, 
                                   param_grid= hyperparam_grid,
                                   cv=StratifiedKFold(5), # Stratified 4 fold cross-validation
                                   n_jobs=-1, 
                                   scoring='recall', 
                                   verbose=2).fit(X_train, y_train) 


# In[51]:


# "Best" Parameters - "Recall"
print(f'Best recall: {gridsearch_balanced.best_score_} with param: {gridsearch_balanced.best_params_}')


# * Recall is almost 73% which is better than other parameter

# ####  Predict and evaluate using the "Best Estimators".

# * a) Use the best estimator from the grid search to make predictions on the "test" set.
# 
# * b) What is the "Recall" on the "test" set for the toxic comments?
# 
# * c) What is the "f_1 score"?.

# In[52]:


# Instantiate "Logistic Regression" from sklearn with "Best Estimators and class weight"

lr_best_class_weights = LogisticRegression(solver='liblinear' , 
                                           class_weight={0: 0.12, 1: 0.88} ,
                                            
                                           C=17 , 
                                           penalty='l2' ,
                                           verbose=2)


# In[53]:


# Fit into  the "train" set.

lr_best_class_weights.fit(X_train, y_train)


# In[54]:


# To make predictions on the "test" set.
y_test_predict_best_class_weights = lr_best_class_weights.predict(X_test)


# In[55]:


# Accuracy of the "test" set
print("Accuracy on the Test Dataset : ", accuracy_score(y_test , y_test_predict_best_class_weights))


# In[56]:


# Recall of the "test" set

print("Recall on the Test Dataset : ", recall_score(y_test , y_test_predict_best_class_weights))


# In[57]:


# F1 - score of the "test" set

print("f1 - score on the Test Dataset : ", f1_score(y_test , y_test_predict_best_class_weights))


# In[58]:


# To make predictions on the "train" set.

y_train_predict_best_class_weights = lr_best_class_weights.predict(X_train)


# In[59]:


# Classification Report - "Train" set

print(classification_report(y_train, y_train_predict_best_class_weights))


# In[60]:


# Classification Report - "Test" set

print(classification_report(y_test, y_test_predict_best_class_weights))


# #### "ROC" (Area Under Receiver Operating Characteristic) Curve --- "Best Estimators and class weight"

# In[61]:


# Calculate fpr, tpr and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_test_predict_best_class_weights)

# Calculate roc_auc
roc_auc = auc(fpr, tpr)

# Plot the figure in the notebook
plt.figure(figsize=(12,8))

# Plot ROC curve
plt.plot(fpr, tpr, color='red', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
# Plot line with no predictive power (baseline)
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='No Skill')

# Set the "x-limits" and "y-limits" of the current axes.
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# Customize the plot
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - "Best Estimators"')
plt.legend(loc="lower right")
plt.show()


# #### Area under ROC Curve

# In[62]:


roc_auc_score(y_test, y_test_predict_best_class_weights)

print(f"AUC_GridSearchCV: {roc_auc_score(y_test, y_test_predict_best_class_weights) * 100:.2f}%")


# ### Conclusion

#                 Summary - "TEST" Data                                      Recall [%]
#             1. L.R with Default Parameters                                      31%
#             
#             2. L.R with Adjusted Class weight Parameters                        62%
# 
#             3. L.R with Best Estimators and Class weight Parameters             71%
#             
#             
#             
# 
# 
#                Summary - "TEST" Data                                       AUC (Area under the Curve) score[%]
#              1. L.R with Default Parameters                                         66.61%                          
# 
#              2. L.R with Adjusted Class weight Parameters                           79.87%
#              
#              3. L.R with Best Estimators and Class weight Parameters                 83.99%
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[ ]:




