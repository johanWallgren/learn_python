# Article: https://medium.com/@stallonejacob/time-series-forecast-a-basic-introduction-using-python-414fcb963000
# Code: https://github.com/jacobstallone/Time_Series_ARIMA--Blog-and-code-
#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

#%%

# Load data
df_raw = pd.read_csv(os.getcwd() + '/airpassengers/airpassengers.csv', index_col=False)
print(df_raw.head())
print(df_raw.dtypes)

#%%
df = df_raw.copy()

# Convert to time series, month column as index
com = df['Month']
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

print(df.index)

# Make it into a series, with month as index
ts = df['#Passengers']
ts.head()
#%%
# Properties of a date_time index

# Two ways of accesing #Passengers
print(ts['1949-01-01'])
print(ts[datetime(1949,1,1)])

# Get a range of values
print(ts['1949-01-01':'1949-05-01'])
print(ts[:'1949-05-01'])

# All values for one year
print(ts['1949'])

#%%
''' STATIONARITY 
A time series is stationary if:
¤ constant mean
¤ constant variance
¤ an auto co-variance that does not depend on time
'''
plt.plot(ts)
print('It’s clear from the plot that there is an overall increase in the trend,with some seasonality in it')

#%%
'''  Dickey fuller Test '''

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
#Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='orange', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

# Run test
test_stationarity(ts)
'''
This is not stationary because :
• mean is increasing even though the std is small.
• Test stat is > critical value.
• Note: the signed values are compared and the absolute values.
'''

#%%
''' MAKING THE TIME SERIES STATIONARY 
• Trend: non-constant mean
• Seasonality: Variation at specific time-frames
'''

# Trend: non-constant mean, fix using smoothing

# Log the series
ts_log = np.log(ts)
plt.plot(ts_log)

#%%
# Moving average
moving_avg = ts_log.rolling(12).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color='red')

#%%
# Remove rolling mean 
ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.dropna(inplace=True) # NA's are introduced becouse of moving average
print(ts_log_moving_avg_diff.head(10))

test_stationarity(ts_log_moving_avg_diff)
'''
• The rolling values are varying slightly but there is no specific trend.
• The test statistics is smaller than the 5 percent critical values. 
That tells us that we are 95 percent confident that this series is stationary.
'''
#%%
# Exponentially weighted moving average

expweighted_avg = ts_log.ewm(halflife=12).mean()
plt.plot(ts_log)
plt.plot(expweighted_avg, color='red')

#%%
ts_log_ewm_avg_diff = ts_log - expweighted_avg
test_stationarity(ts_log_ewm_avg_diff)

'''
It is stationary because:
• Rolling values have less variations in mean and standard deviation in magnitude.
• the test statistic is smaller than 1 percent of the critical value. 
So we can say we are almost 99 percent confident that this is stationary.
'''

#%%
''' Seasonality (along with Trend) '''
# Differencing

ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)

plt.plot(ts_log_diff)
#%%
test_stationarity(ts_log_diff)

'''
It is stationary because:
• the mean and std variations have small variations with time.
• test statistic is less than 10 percent of the critical values, so we can be 90 percent confident that this is stationary.
'''
#%%

# Decomposing
decomp = seasonal_decompose(ts_log)
trend = decomp.trend
seasonal = decomp.seasonal
residual = decomp.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residual')
plt.legend(loc='best')
#%%

ts_log_decomp = residual
ts_log_decomp.dropna(inplace=True)
test_stationarity(ts_log_decomp)

'''
This is stationary because:
• test statistic is lower than 1 percentile critical values.
• the mean and std variations have small variations with time.
'''
#%%
''' Forecasting a Time Series using ARIMA 
• p : This is the number of AR (Auto-Regressive) terms . Example — if p is 3 the predictor for y(t) will be y(t-1),y(t-2),y(t-3).
• q : This is the number of MA (Moving-Average) terms . Example — if p is 3 the predictor for y(t) will be y(t-1),y(t-2),y(t-3).
• d :This is the number of differences or the number of non-seasonal differences .
'''
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')

'''
• p: The first time where the PACF crosses the upper confidence interval, here its close to 2. hence p = 2.
• q: The first time where the ACF crosses the upper confidence interval, here its close to 2. hence p = 2.
'''
# MA model
#%%
model = ARIMA(ts_log, order=(2,1,0))
results_AR = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues - ts_log_diff)**2))


#%%
model = ARIMA(ts_log, order=(0,1,2))
results_MA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues - ts_log_diff)**2))

#%%
model = ARIMA(ts_log, order=(2,1,2))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues - ts_log_diff)**2))

#%%
''' FINAL STEP: BRINGING THIS BACK TO THE ORIGINAL SCALE 
• First get the predicted values and store it as series. 
You will notice the first month is missing because we took a lag of 1(shift).
• Now convert differencing to log scale: find the cumulative sum and add it to a new 
series with a base value (here the first-month value of the log series).
'''

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())

predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
print(predictions_ARIMA_log.head())

#%%
''' • Next -take the exponent of the series from above (anti-log) which will be the predicted value — the time series forecast model.'''

prediction_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(prediction_ARIMA)
plt.title('RSS: %.4f'% sum((prediction_ARIMA - ts_log_diff)**2))
