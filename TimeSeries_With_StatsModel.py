#!/usr/bin/env python
# coding: utf-8

# # Introduction to Statsmodels
# 
# Statsmodels is a Python module that provides classes and functions for the estimation of many different statistical models, as well as for conducting statistical tests, and statistical data exploration. An extensive list of result statistics are available for each estimator. The results are tested against existing statistical packages to ensure that they are correct. The package is released under the open source Modified BSD (3-clause) license. The online documentation is hosted at <a href='https://www.statsmodels.org/stable/index.html'>statsmodels.org</a>. The statsmodels version used in the development of this course is 0.9.0.

# In[2]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('macrodata.csv',index_col=0,parse_dates=True)
df.head()


# ### Perform standard imports and load the dataset
# For these exercises we'll be using a statsmodels built-in macroeconomics dataset:
# 
# <pre><strong>US Macroeconomic Data for 1959Q1 - 2009Q3</strong>
# Number of Observations - 203
# Number of Variables - 14
# Variable name definitions:
#     year      - 1959q1 - 2009q3
#     quarter   - 1-4
#     realgdp   - Real gross domestic product (Bil. of chained 2005 US$,
#                 seasonally adjusted annual rate)
#     realcons  - Real personal consumption expenditures (Bil. of chained
#                 2005 US$, seasonally adjusted annual rate)
#     realinv   - Real gross private domestic investment (Bil. of chained
#                 2005 US$, seasonally adjusted annual rate)
#     realgovt  - Real federal consumption expenditures & gross investment
#                 (Bil. of chained 2005 US$, seasonally adjusted annual rate)
#     realdpi   - Real private disposable income (Bil. of chained 2005
#                 US$, seasonally adjusted annual rate)
#     cpi       - End of the quarter consumer price index for all urban
#                 consumers: all items (1982-84 = 100, seasonally adjusted).
#     m1        - End of the quarter M1 nominal money stock (Seasonally
#                 adjusted)
#     tbilrate  - Quarterly monthly average of the monthly 3-month
#                 treasury bill: secondary market rate
#     unemp     - Seasonally adjusted unemployment rate (%)
#     pop       - End of the quarter total population: all ages incl. armed
#                 forces over seas
#     infl      - Inflation rate (ln(cpi_{t}/cpi_{t-1}) * 400)
#     realint   - Real interest rate (tbilrate - infl)</pre>
#     
# <div class="alert alert-info"><strong>NOTE:</strong> Although we've provided a .csv file in the Data folder, you can also build this DataFrame with the following code:<br>
# <tt>&nbsp;&nbsp;&nbsp;&nbsp;import pandas as pd<br>
# &nbsp;&nbsp;&nbsp;&nbsp;import statsmodels.api as sm<br>
# &nbsp;&nbsp;&nbsp;&nbsp;df = sm.datasets.macrodata.load_pandas().data<br>
# &nbsp;&nbsp;&nbsp;&nbsp;df.index = pd.Index(sm.tsa.datetools.dates_from_range('1959Q1', '2009Q3'))<br>
# &nbsp;&nbsp;&nbsp;&nbsp;print(sm.datasets.macrodata.NOTE)</tt></div>

# In[3]:


# Plot


# In[4]:


ax = df['realgdp'].plot()
ax.autoscale(axis='x',tight=True)
ax.set(ylabel='REAL GDP')


# In[5]:


## Analyzing the Trend


# In[6]:


from statsmodels.tsa.filters.hp_filter import hpfilter
# Tuple unpacking
gdp_cycle, gdp_trend = hpfilter(df['realgdp'])


# In[7]:


df['trend'] = gdp_trend


# In[6]:


df[['trend','realgdp']].plot().autoscale(axis='x',tight=True)


# In[8]:


df['gdp_cycle'] =gdp_cycle
df[['gdp_cycle','realgdp']].plot().autoscale(axis='x',tight=True)


# # Method 2
# # ETS DECOMPOSITION
# # Error/Trend/Seasonality Models

# In[9]:


airline = pd.read_csv('airline_passengers.csv',index_col='Month',parse_dates=True)
airline.head()


# In[9]:


airline.plot()


# In[10]:


cyc, trend = hpfilter(airline['Thousands of Passengers'])
airline['trend'] = trend
airline['cyclic'] = cyc


# In[11]:


airline[['trend','Thousands of Passengers']].plot().autoscale(axis='x',tight=True)


# In[13]:


airline[['cyclic','Thousands of Passengers']].plot().autoscale(axis='x',tight=True)


# In[15]:


from statsmodels.tsa.seasonal import seasonal_decompose
airline = pd.read_csv('airline_passengers.csv',index_col='Month',parse_dates=True)
result = seasonal_decompose(airline['Thousands of Passengers'], model='multiplicative')  # model='mul' also works

result.plot();


# # IMAGE OF ETS - DECOMOPOSITION

# ![image.png](attachment:image.png)

# # Simple Moving Average

# In[16]:


airline = pd.read_csv('airline_passengers.csv',index_col='Month',parse_dates=True)
airline.dropna(inplace=True)
airline['6-month-SMA'] = airline['Thousands of Passengers'].rolling(window=6).mean()
airline['12-month-SMA'] = airline['Thousands of Passengers'].rolling(window=12).mean()


# In[17]:


airline.head(15)


# In[18]:


# Evaluate


# In[19]:


airline.plot();


# # Exponentially Weighted Moving Average

# In[29]:


airline['EWMA12'] = airline['Thousands of Passengers'].ewm(span=12,adjust=False).mean()
airline[['Thousands of Passengers','EWMA12']].plot();


# # Compare SMA vs EWMA

# In[30]:


airline[['Thousands of Passengers','EWMA12','12-month-SMA']].plot(figsize=(12,8)).autoscale(axis='x',tight=True);


# ## Holt-Winters Methods

#     Double Smoothing -We add Trend as Input parameter on top of EMA
#     Triple Smoothing - We add Trend and Seasonality as Input Parameter

# In[23]:


df = pd.read_csv('airline_passengers.csv',index_col='Month',parse_dates=True)
df.dropna(inplace=True)
df.head()


# In order to build a Holt-Winters smoothing model, statsmodels needs to know the frequency of the data (whether it's daily, monthly etc.). Since observations occur at the start of each month, we'll use MS

# In[22]:


df.index.freq = 'MS'
df.head()


# In[31]:


from statsmodels.tsa.holtwinters import SimpleExpSmoothing
span = 12
alpha = 2/(span+1)

df['EWMA12'] = df['Thousands of Passengers'].ewm(alpha=alpha,adjust=False).mean()
df['SES12']=SimpleExpSmoothing(df['Thousands of Passengers']).fit(smoothing_level=alpha,optimized=False).fittedvalues.shift(-1)
df.head()


# # Double Exponential Smoothing

# Where Simple Exponential Smoothing employs just one smoothing factor <Strong> 𝛼  </Strong>(alpha), Double Exponential Smoothing adds a second smoothing factor <Strong> 𝛽 </Strong> (beta) that addresses trends in the data
# 
# We can also address different types of change (growth/decay) in the trend. If a time series displays a straight-line sloped trend, you would use an <Strong>additive </Strong>adjustment. If the time series displays an exponential (curved) trend, you would use a <Strong> multiplicative </Strong>adjustment.

# In[36]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing
df['DESadd12'] = ExponentialSmoothing(df['Thousands of Passengers'], trend='add').fit().fittedvalues.shift(-1)
df.head()


# In[37]:


df[['Thousands of Passengers','EWMA12','DESadd12']].iloc[:24].plot(figsize=(12,6)).autoscale(axis='x',tight=True);


# In[38]:


## Double Exponential is much better than other


# In[39]:


df['DESmul12'] = ExponentialSmoothing(df['Thousands of Passengers'], trend='mul').fit().fittedvalues.shift(-1)
df.head()


# In[40]:


df[['Thousands of Passengers','DESadd12','DESmul12']].iloc[:24].plot(figsize=(12,6)).autoscale(axis='x',tight=True);


# MUL is better than ADD in this case

# ___
# ## Triple Exponential Smoothing
# Triple Exponential Smoothing, the method most closely associated with Holt-Winters, adds support for both trends and seasonality in the data. 
# 

# In[41]:


df['TESadd12'] = ExponentialSmoothing(df['Thousands of Passengers'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
df.head()


# In[42]:


import warnings
warnings.filterwarnings('ignore')


# In[43]:


df['TESadd12'] = ExponentialSmoothing(df['Thousands of Passengers'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
df.head()


# In[44]:


df['TESmul12'] = ExponentialSmoothing(df['Thousands of Passengers'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues
df.head()


# In[45]:


df[['Thousands of Passengers','TESadd12','TESmul12']].plot(figsize=(12,6)).autoscale(axis='x',tight=True);


# In[46]:


## Take a Subset
df[['Thousands of Passengers','TESadd12','TESmul12']].iloc[:24].plot(figsize=(12,6)).autoscale(axis='x',tight=True);


# # The most imp aspect of Time Series is to able to predict

# # Forecasting with the Holt-Winters Method

# In[47]:


df = pd.read_csv('airline_passengers.csv',index_col='Month',parse_dates=True)
df.index.freq = 'MS'
df.head()


# In[48]:


df.shape


# # Train Test Split

# In[49]:


train_data = df.iloc[:108] # Goes up to but not including 108
test_data = df.iloc[108:]


# In[50]:


print("Size of Testing Data set is", 144-108)


# # Build the Model

# In[51]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing
fitted_model = ExponentialSmoothing(train_data['Thousands of Passengers'],trend='mul',seasonal='mul',seasonal_periods=12).fit()


# In[52]:


y_pred = fitted_model.forecast(36)


# In[53]:


import pandas as pd
ndf = pd.DataFrame()


# In[54]:


ndf['predicted'] = y_pred
ndf['actualvalue'] = test_data['Thousands of Passengers']
ndf.head()


# In[55]:


ndf['diff'] = ndf['actualvalue'] - ndf['predicted']
ndf.head()


# In[56]:


ndf.tail()


# In[57]:


y_pred_future = fitted_model.forecast(40)


# In[58]:


y_pred_future.tail()


# # PREDICT

# In[59]:


test_predictions = fitted_model.forecast(36).rename('HW Forecast')


# In[60]:


test_predictions


# In[61]:


train_data['Thousands of Passengers'].plot(legend=True,label='TRAIN')
test_data['Thousands of Passengers'].plot(legend=True,label='TEST',figsize=(12,8));


# In[62]:


## Print Predicted Data along with TEST Data
train_data['Thousands of Passengers'].plot(legend=True,label='TRAIN')
test_data['Thousands of Passengers'].plot(legend=True,label='TEST',figsize=(12,8));
test_predictions.plot(legend=True,label='PREDICTION');


# # Evaluation Metrics

# In[63]:


from sklearn.metrics import mean_squared_error,mean_absolute_error
mae = mean_absolute_error(test_data,test_predictions)
mse = mean_squared_error(test_data,test_predictions)
RMSE = np.sqrt(mean_squared_error(test_data,test_predictions))


# In[64]:


print("Mean Absolute Error is", mae)
print("Mean Sqr Error is", mse)
print("RMSE ", RMSE)


# # FORECAST Usng Model

# In[65]:


final_model = ExponentialSmoothing(df['Thousands of Passengers'],trend='mul',seasonal='mul',seasonal_periods=12).fit()


# In[66]:


forecast_predictions = final_model.forecast(36)


# # STATIONARY

# In[70]:


df2 = pd.read_csv('samples.csv',index_col=0,parse_dates=True)
df2.head()


# In[ ]:


df2['a'].plot(ylim=[0,100],title="STATIONARY DATA").autoscale(axis='x',tight=True);


# In[68]:


df2['b'].plot(ylim=[0,100],title="NON-STATIONARY DATA").autoscale(axis='x',tight=True);


# In[69]:


df2['c'].plot(ylim=[0,10000],title="MORE NON-STATIONARY DATA").autoscale(axis='x',tight=True);


# # Differencing
# ## First Order Differencing
# Non-stationary data can be made to look stationary through <em>differencing</em>. A simple method called <em>first order differencing</em> calculates the difference between consecutive observations.
# 
# &nbsp;&nbsp;&nbsp;&nbsp;$y^{\prime}_t = y_t - y_{t-1}$
# 
# In this way a linear trend is transformed into a horizontal set of values.
# 

# In[62]:


# Calculate the first difference of the non-stationary dataset "b"
df2['d1b'] = df2['b'] - df2['b'].shift(1)

df2[['b','d1b']].head()


# In[63]:


df2['d1b'].plot(title="FIRST ORDER DIFFERENCE").autoscale(axis='x',tight=True);


# In[214]:


df2['b'].plot(ylim=[0,100],title="Without Difference").autoscale(axis='x',tight=True);


# In[67]:


## An easier way to perform differencing on a pandas Series or DataFrame is to use the built-in .diff() method:
df2['d1bb'] = df2['b'].diff()
df2['d1bb'].plot(title="FIRST ORDER DIFFERENCE USING PANDAS").autoscale(axis='x',tight=True);


# ### Forecasting on first order differenced data
# When forecasting with first order differences, the predicted values have to be added back in to the original values in order to obtain an appropriate forecast.
# 
# Let's say that the next five forecasted values after applying some model to <tt>df['d1b']</tt> are <tt>[7,-2,5,-1,12]</tt>. We need to perform an <em>inverse transformation</em> to obtain values in the scale of the original time series.

# In[68]:


# For our example we need to build a forecast series from scratch
# First determine the most recent date in the training set, to know where the forecast set should start
df2[['b']].tail(5)


# In[69]:


# Next set a DateTime index for the forecast set that extends 5 periods into the future
idx = pd.date_range('1960-01-01', periods=5, freq='MS')
idx


# In[70]:


# Create a Z dataframe
z = pd.DataFrame([7,-2,5,-1,12],index=idx,columns=['Fcast'])
z


# In[71]:


df2['b'].iloc[-1]


# In[232]:


z['Fcast']


# In[72]:


z['cumsum'] = z['Fcast'].cumsum()


# In[234]:


z


# The idea behind an inverse transformation is to start with the most recent value from the training set, and to add a cumulative sum of Fcast values to build the new forecast set. For this we'll use the pandas <tt>.cumsum()</tt> function which does the reverse of <tt>.diff()</tt>

# In[73]:


z['forecast']=df2['b'].iloc[-1] + z['Fcast'].cumsum()
z


# In[237]:


# 7 + 73 = 80
# 5 + 73 = 78
# 10 + 73 = 83


# In[74]:


df2['b'].plot(figsize=(12,5), title="FORECAST").autoscale(axis='x',tight=True)

z['forecast'].plot();


# ## Second order differencing
# Sometimes the first difference is not enough to attain stationarity, particularly if the trend is not linear. We can difference the already differenced values again to obtain a second order set of values.
# 
# &nbsp;&nbsp;&nbsp;&nbsp;$\begin{split}y_{t}^{\prime\prime} &= y_{t}^{\prime} - y_{t-1}^{\prime} \\
# &= (y_t - y_{t-1}) - (y_{t-1} - y_{t-2}) \\
# &= y_t - 2y_{t-1} + y_{t-2}\end{split}$

# In[239]:


# First we'll look at the first order difference of dataset "c"
df2['d1c'] = df2['c'].diff()
df2['d1c'].plot(title="FIRST ORDER DIFFERENCE").autoscale(axis='x',tight=True);


# Now let's apply a second order difference to dataset "c".

# In[240]:


# We can do this from the original time series in one step
df2['d2c'] = df2['c'].diff().diff()

df2[['c','d1c','d2c']].head()


# In[241]:


df2['d2c'].plot(title="SECOND ORDER DIFFERENCE").autoscale(axis='x',tight=True);


# # ACF and PACF

# # Autocorrelation Function / Partial Autocorrelation Function

# In[242]:


# Load a non-stationary dataset
df1 = pd.read_csv('airline_passengers.csv',index_col='Month',parse_dates=True)
df1.index.freq = 'MS'

# Load a stationary dataset
df2 = pd.read_csv('DailyTotalFemaleBirths.csv',index_col='Date',parse_dates=True)
df2.index.freq = 'D'


# In[243]:


df1.plot(title="Airlline Passengers").autoscale(axis='x',tight=True);
df2.plot(title="Daily Tota lFemale Births").autoscale(axis='x',tight=True);


# In[244]:


# Import the models we'll be using in this section
from statsmodels.tsa.stattools import acovf,acf,pacf,pacf_yw,pacf_ols


# In[245]:


df = pd.DataFrame({'a':[13, 5, 11, 12, 9]})
df


# In[246]:


acf(df['a'])


# In[247]:


pacf(df['a'])


# In[248]:


from pandas.plotting import lag_plot
lag_plot(df1['Thousands of Passengers']);


# In[249]:


lag_plot(df2['Births']);


# # Airline Passenger shows Autocorrelation vs Births 

# In[250]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# # ACF

# In[251]:


acf(df2['Births'])


# In[252]:


# Now let's plot the autocorrelation at different lags
title = 'Autocorrelation: Daily Female Births'
lags = 40
plot_acf(df2,title=title,lags=lags);


# In[253]:


title = 'Autocorrelation: Airline Passengers'
lags = 40
plot_acf(df1,title=title,lags=lags);


# # PACF

# In[254]:


title='Partial Autocorrelation: Daily Female Births'
lags=40
plot_pacf(df2,title=title,lags=lags);


# In[255]:


title = 'Partial Autocorrelation: Airline Passengers'
lags = 40
plot_pacf(df1,title=title,lags=lags);


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




