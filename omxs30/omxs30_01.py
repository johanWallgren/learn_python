#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from pandas.plotting import autocorrelation_plot

#%%
def get_omxs30_averages(company_data_file, LAG):
    # Function that returns a data frame with index=date and the daily average stock price
    # Replaces column name with stock abrivation
    stock_name, _ = company_data_file.split('-', 1)

    df_raw = pd.read_csv(os.getcwd() + '\\data\\' + company_data_file, 
        index_col=False, sep=';', skiprows=1, decimal=',', parse_dates=[0])

    df_raw.set_index('Date', inplace=True)
    df = df_raw[['Average price']]
    df = df.diff()
    df = df.rolling(LAG).mean()
    df = df.dropna()
    df.columns = [stock_name]
    return(df)

# All files 
LAG = 30

omxs30_files = [f for f in os.listdir(os.getcwd() + '\data') if os.path.isfile(os.path.join(os.getcwd() + '\data', f))]

df = get_omxs30_averages(omxs30_files[0], LAG)
for f in omxs30_files[1:]:
    df = df.join(get_omxs30_averages(f, LAG), on='Date', how='left')

# Create lag for each stock and find best corr stock

df_lag = df.copy()

stocks = list(df_lag.columns)
max_corr_stock = []
max_corr_value = []

for i, f in enumerate(list(df_lag.columns)):
    df_lag['lag'] = df_lag[f].shift(LAG)
    dfcorr = df_lag.corr().abs().sort_values(by='lag', ascending=False)
    dfcorr = dfcorr.loc[~dfcorr.index.isin(['lag', f])] # Removes stock and lag
    max_corr_stock.append(dfcorr.index[0]) # Stock with highest corr is now in row 0
    max_corr_value.append(dfcorr.iloc[0,-1]) # Corr value between stock and stock with highest corr is in last column

results_stock_corr = pd.DataFrame({'stocks':stocks, 
'max_corr_stock':max_corr_stock, 
'max_corr_value':max_corr_value}).sort_values(by='max_corr_value',ascending=False)

results_stock_corr.head(10)

#%%
df_HM_GETI = df.copy()

df_HM_GETI['lag'] = df_lag['GETI'].shift(LAG)
df_HM_GETI[['HM', 'lag']].plot()



#%%
