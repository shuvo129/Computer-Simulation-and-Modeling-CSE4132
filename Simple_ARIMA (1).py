#!/usr/bin/env python
# coding: utf-8

# In[21]:


##  Step 1: Declare the necessary libraries
    
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_predict


def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')
 
##  Step 2(a): Input Data  
    
File_Input_Path = "C:/Users/asifz/OneDrive/Desktop/Class/"
Input_File_Name = File_Input_Path + "shampoo" + ".csv"

series = read_csv(Input_File_Name, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
#print(series.head(20))

## Step 2(b): Data Preparation Make the data non-seasonal 
#### Concentrated on Mean line without too much Variance
#### Seems that data is seasonal (Not concentrated on Mean line)
series.plot()
plt.xlabel("Time: Month")
plt.ylabel("Sales Figures")
plt.title("Raw Data Plot")
plt.show()

print("Seems that data is seasonal. We have to make it non seasonal")
'''
We can see that the Shampoo Sales dataset has a clear trend.
This suggests that the time series is not stationary and 
will require differencing to make it stationary, 
at least a difference order of 1.
'''

## Step 3: Model Identification
# ACF plot
autocorrelation_plot(series)
plt.xlabel("LAG")
plt.ylabel("ACF Values")
plt.title("ACF (Autocorrelation)  Plot")
plt.show()

print("Seems that correlation value start to sharp decay at lag 5")

"""
We can see that there is a positive correlation with the first 10-to-12 lags 
that is perhaps significant for the first 5 lags.
A good starting point for the AR parameter of the model may be 5.

First, we fit an ARIMA(5,1,0) model. 

Here ARIMA(Lag p = 5, 
           Difference d = 1,  
           Moving Avg. q = 0).

This sets the lag value, p to 5 for autoregression, 
uses a difference order, d of 1 to make the time series stationary, 
and uses a moving average, q model of 0.
"""

## Step 4: Create an ARIMA model AR(5,1,0) 

model = ARIMA(series, order=(5,1,0))

## Step 5: Fit the model

model_fit = model.fit()

print(model_fit.summary())
"""
Running the example prints a summary of the fit model. 
This summarizes the coefficient values used as well as the 
skill of the fit on the in-sample observations.
"""


### ARIMA_model_1.py
# Actual vs Fitted
plot_predict(model_fit, dynamic=False)
plt.show()


### Checking for any trend 

"""
First: 
See a line plot of the residual errors, suggesting that there may 
still be some trend information not captured by the model.
"""
###  Plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.xlabel("Time: Month")
plt.ylabel("Residual Values")
plt.title("ARIMA Fit Residuals Error line Plot")
plt.show()


"""
Second: 
Check with a density plot of the residual error values, 
suggesting the errors are Gaussian, but may not be centered on zero.
"""



###  First, we get a line plot of the residual errors, suggesting that there may still be some trend 
###  information not captured by the model.

"""
count   35.000000
mean    -5.495213
std     68.132882
min   -133.296597
25%    -42.477935
50%     -7.186584
75%     24.748357
max    133.237980

The results show that indeed there is a bias in the prediction 
(a non-zero mean in the residuals).
"""

### Step 6: Using the fitted model forcast the new values
###  Next, we get a density plot of the residual error values, suggesting the 
###  errors are Gaussian, but may not be centered on zero.

X = series.values
print('Actual data Size = ', len(X))

size = int(len(X) * 0.66)


print('Train data Size = ', int(len(X) * 0.66))

print('Train data Size = ', int(len(X) * 0.34))


## Split the dataset into train ((36 observations)) and test (12 observations)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('Predicted=%f, (Actual Values) Expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# plot
plt.plot(test, color='green', label="Actual Value")
plt.plot(predictions, color='red', label="Predicted Value")
plt.legend()
plt.xlabel("No. of Observations")
plt.ylabel("Sales Figure")
plt.title("Actual VS Predicted")
plt.show()


# In[1]:


##  Step 1: Declare the necessary libraries
    
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_predict


# In[2]:


def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')
 
##  Step 2(a): Input Data  
    
File_Input_Path = "C:/Users/asifz/OneDrive/Desktop/Class/"
Input_File_Name = File_Input_Path + "shampoo" + ".csv"

series = read_csv(Input_File_Name, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
#print(series.head(20))

## Step 2(b): Data Preparation Make the data non-seasonal 
#### Concentrated on Mean line without too much Variance
#### Seems that data is seasonal (Not concentrated on Mean line)
series.plot()
plt.xlabel("Time: Month")
plt.ylabel("Sales Figures")
plt.title("Raw Data Plot")
plt.show()


# In[3]:


## Step 3: Model Identification
# ACF plot
autocorrelation_plot(series)
plt.xlabel("LAG")
plt.ylabel("ACF Values")
plt.title("ACF (Autocorrelation)  Plot")
plt.show()

print("Seems that correlation value start to sharp decay at lag 5")


# In[4]:


## Step 4: Create an ARIMA model AR(5,1,0) 

model = ARIMA(series, order=(5,1,0))

## Step 5: Fit the model

model_fit = model.fit()

print(model_fit.summary())


# In[5]:


### ARIMA_model_1.py
# Actual vs Fitted
plot_predict(model_fit, dynamic=False)
plt.show()


### Checking for any trend 

"""
First: 
See a line plot of the residual errors, suggesting that there may 
still be some trend information not captured by the model.
"""
###  Plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.xlabel("Time: Month")
plt.ylabel("Residual Values")
plt.title("ARIMA Fit Residuals Error line Plot")
plt.show()


# In[6]:


X = series.values
print('Actual data Size = ', len(X))

size = int(len(X) * 0.66)


print('Train data Size = ', int(len(X) * 0.66))

print('Train data Size = ', int(len(X) * 0.34))


# In[7]:


## Split the dataset into train ((36 observations)) and test (12 observations)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('Predicted=%f, (Actual Values) Expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)


# In[8]:


# plot
plt.plot(test, color='green', label="Actual Value")
plt.plot(predictions, color='red', label="Predicted Value")
plt.legend()
plt.xlabel("No. of Observations")
plt.ylabel("Sales Figure")
plt.title("Actual VS Predicted")
plt.show()


# In[ ]:




