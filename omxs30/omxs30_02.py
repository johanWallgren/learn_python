#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from pandas.plotting import autocorrelation_plot
from enum import unique
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#%%
def get_data_csv_to_df(csv_file_name, PATH_TO_OMXS30_CSV_FILES):
    # Function that returns a data frame with index=date and the daily closing stock price
    # Replaces column name with stock abrivation
    stock_name, _ = csv_file_name.split('-', 1) # Extract stock name from csv file name

    df_one_stock = pd.read_csv(PATH_TO_OMXS30_CSV_FILES + csv_file_name, 
        index_col=False, sep=';', skiprows=1, decimal=',', parse_dates=[0])

    df_one_stock.set_index('Date', inplace=True) # Set date as index
    df_one_stock = df_one_stock[['Closing price']] # Select closing price only
    df_one_stock.columns = [stock_name] # Rename column to stock name
    return(df_one_stock)


def calc_row_diff_and_5day_mean(df_raw, roll_days):
    df = df_raw.copy().sort_values(by='Date', ascending=True)
    df = df.diff(axis=0).rolling(roll_days, axis=0).mean().dropna()
    return(df)


def calc_lag_and_cut_time_period(df, lag):
    # Lags df and then adjusts the length of df
    # df_lag contains the predictor values
    # df_adjusted_for_lag contatins the response values
    date_column = df.index.copy()

    df_lag = df.copy()
    df_lag = df_lag[lag:] # Remove top rows
    df_lag['Date'] = date_column[:-lag] # Remove last rows from date column
    df_lag.set_index('Date', inplace=True) # Replace index with the earlier dates

    df_adjusted_for_lag = df.copy()
    df_adjusted_for_lag = df_adjusted_for_lag[:-lag] # Removes last rows so that df's has same length

    return(df_lag, df_adjusted_for_lag)

#%%
# Load data and perform corr analysis between all stocks for different lags
# Save significant (>MIN_CORR_VALUE) results to dataframe (results_stock_corr)

PATH_TO_OMXS30_CSV_FILES = os.getcwd() + '\\data\\omxs30\\' # Directory for omxs30 csv files
DAYS_ROLLING = 5
MIN_CORR_VALUE = 0.23

omxs30_csv_files = [f for f in os.listdir(PATH_TO_OMXS30_CSV_FILES) if os.path.isfile(os.path.join(PATH_TO_OMXS30_CSV_FILES, f))]

# Load omxs30 csv files to df_raw
df_raw = get_data_csv_to_df(omxs30_csv_files[0], PATH_TO_OMXS30_CSV_FILES)
for f in omxs30_csv_files[1:]:
    df_raw = df_raw.join(get_data_csv_to_df(f, PATH_TO_OMXS30_CSV_FILES), on='Date', how='left')

# Calculate diff between days and then the 5 day rolling average
df = calc_row_diff_and_5day_mean(df_raw, DAYS_ROLLING)

# Test lags from DAYS_ROLLING + 1 to 31 days
results_stock_corr = pd.DataFrame(columns=['stock','corr_stock','lag','corr'])

for lag in range(DAYS_ROLLING + 1, 32):
    # Loop over all lag's

    df_lag, df_adjusted_for_lag = calc_lag_and_cut_time_period(df, lag)

    for i, f in enumerate(list(df_adjusted_for_lag.columns)):
        # Loop over all OMXS30 stocks

        response_col = df_adjusted_for_lag[[f]]
        response_col.columns = ['response']
        df_corr = response_col.join(df_lag, on='Date', how='left').corr()
        df_corr = df_corr[['response']]
        df_corr = df_corr.loc[~df_corr.index.isin(['response'])]

        for corr_stock, corr in enumerate(df_corr['response']):
            # Loop over all correlations and write to results_stock_corr

            if abs(corr) > MIN_CORR_VALUE:
                # If corr is greater than MIN_CORR_VALUE, add the data to dataframe results_stock_corr
                # print(f + ' - ' + df_corr.index[corr_stock] + ' @ lag:' + str(lag) + ' corr: ' + str(corr))
                results_stock_corr = results_stock_corr.append(
                    {'stock':f,
                    'corr_stock':df_corr.index[corr_stock],
                    'lag':lag,
                    'corr':corr},
                    ignore_index=True)


# Sort and present results
results_stock_corr.sort_values(by=['stock','lag'], inplace=True)
results_stock_corr.head(25)

#%%
# Build a dataframe for each OMXS30 stock with df_adjusted_for_lag['stock'] and all predictions columns calculated using their lag
# Ex: df_adjusted_for_lag['SEB'], df_lag['BOL']@lag=24, df_lag['BOL']@lag=25, df_lag['SWED']@lag=25

stocks_with_corr = list(set(results_stock_corr['stock']))
 
for stock in stocks_with_corr:  
    corr_data_one_stock = results_stock_corr[results_stock_corr['stock'] == stock]
    df_for_prediction = df_adjusted_for_lag[[stock]]

    for row in range(0,len(corr_data_one_stock)):
        predictor, _ = calc_lag_and_cut_time_period(df[[corr_data_one_stock.iloc[row,1]]], corr_data_one_stock.iloc[row,2])
        predictor.columns = [list(df[[corr_data_one_stock.iloc[row,1]]].columns)[0] + '_' + str(corr_data_one_stock.iloc[row,2])]
        df_for_prediction = df_for_prediction.join(predictor, on='Date', how='left')

    split_row = int(len(df_for_prediction) * 0.66)
    train = df_for_prediction[:split_row]
    test = df_for_prediction[split_row:]
    x_train = train.iloc[:, 1:]
    y_train = train.iloc[:,0]
    x_test = test.iloc[:, 1:]
    y_test = test.iloc[:,0]

    ols = LinearRegression()
    model = ols.fit(x_train, y_train)
    ols_prediction = model.predict(x_test)

    print(list(df_for_prediction.columns)[0] + ': RMSE: ' + str(np.sqrt(mean_squared_error(y_test, ols_prediction))) +' R2: ' + str(r2_score(y_test, ols_prediction)))

#%%
