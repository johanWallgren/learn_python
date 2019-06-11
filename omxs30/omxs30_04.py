#%%
# Stock data downloaded from http://www.nasdaqomxnordic.com/aktier/historiskakurser

# Changes with omxs30_03
# Will predict if stock price i equal or higher DAYS_FOR_PREDICTION days later = 1, else = 0
# Function calc_row_diff_and_5day_mean is replaced
# Constant DAYS_ROLLING is removed
# Use different methods for prediction and then vote

# Changes with omxs30_04
# Add new data sources, example, price of steel, other stocks, market index

# To do:
# Set hyperparamethers for all prediction methods

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from pandas.plotting import autocorrelation_plot
from enum import unique
from sklearn.linear_model import LogisticRegression  
from sklearn import svm  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

def get_data_csv_to_df(csv_file_name, path_to_csv_files):
    # Function that returns a data frame with index=date and the daily closing stock price
    # Replaces column name with stock abrivation
    stock_name, _ = csv_file_name.split('-', 1) # Extract stock name from csv file name, error if both A and B stocks are uesd, change file names

    df_one_stock = pd.read_csv(path_to_csv_files + csv_file_name, 
        index_col=False, sep=';', skiprows=1, decimal=',', parse_dates=[0])

    df_one_stock.set_index('Date', inplace=True) # Set date as index
    df_one_stock = df_one_stock[['Closing price']] # Select closing price only
    df_one_stock.columns = [stock_name] # Rename column to stock name
    return(df_one_stock)

def get_if_stock_price_increase(df_raw, DAYS_FOR_PREDICTION):
    # Price is same or greater than DAYS_FOR_PREDICTION earlier = 1, else = 0 

    df_tmp = df_raw.copy().sort_values(by='Date', ascending=True).dropna()
    df = pd.DataFrame(index=df_tmp.index.copy())
    stock_names = list(df_tmp.columns)

    for stock in stock_names: # Loop over all stocks
        one_stock = df_tmp[[stock]].copy()
        one_stock['history'] = one_stock.shift(DAYS_FOR_PREDICTION)
        one_stock.columns = ['today', 'history']
        one_stock.dropna(inplace=True) # Drop rows without history 
        one_stock[stock] = np.where(one_stock['today'] >= one_stock['history'], 1, 0) 

        df = df.join(one_stock[[stock]], on='Date', how='right')

    return(df)


def calc_lag_and_cut_time_period(df, lag):
    # Lags df and then adjusts the length of df
    # df_lag contains the predictor values
    # df_adjusted_for_lag contatins the response values
    # df_adjusted_for_lag and df_lag have the same length

    date_column = df.index.copy()

    df_lag = df.copy()
    df_lag = df_lag[lag:] # Remove top rows
    df_lag['Date'] = date_column[:-lag] # Remove last rows from date column
    df_lag.set_index('Date', inplace=True) # Replace index with the earlier dates

    df_adjusted_for_lag = df.copy()
    df_adjusted_for_lag = df_adjusted_for_lag[:-lag] # Removes last rows so that df's has same length

    return(df_lag, df_adjusted_for_lag)


# Load data and perform corr analysis between all stocks for different lags
# Save significant (>MIN_CORR_VALUE) results to dataframe (results_stock_corr)

# Constants
PATH_TO_OMXS30_CSV_FILES = os.getcwd() + '\\data\\omxs30\\' # Directory for omxs30 csv files
PATH_TO_OTHER_STOCK_CSV_FILES = os.getcwd() + '\\data\\other_stock\\' # Directory for other csv files, only for predictors, not for response
MIN_CORR_VALUE = 0.17
DAYS_FOR_PREDICTION = 5 # How many days in the future to predict
DAYS_FOR_EVALUATION = 7 # How many days in the future to evaluate the models

# Load omxs30 csv files to df_raw
omxs30_csv_files = [f for f in os.listdir(PATH_TO_OMXS30_CSV_FILES) if os.path.isfile(os.path.join(PATH_TO_OMXS30_CSV_FILES, f))]
df_raw = get_data_csv_to_df(omxs30_csv_files[0], PATH_TO_OMXS30_CSV_FILES)
for f in omxs30_csv_files[1:]:
    df_raw = df_raw.join(get_data_csv_to_df(f, PATH_TO_OMXS30_CSV_FILES), on='Date', how='left')

# Load other stock csv files to df_other_raw
other_stock_csv_files = [f for f in os.listdir(PATH_TO_OTHER_STOCK_CSV_FILES) if os.path.isfile(os.path.join(PATH_TO_OTHER_STOCK_CSV_FILES, f))]
df_other_raw = get_data_csv_to_df(other_stock_csv_files[0], PATH_TO_OTHER_STOCK_CSV_FILES)
for f in other_stock_csv_files[1:]:
    df_other_raw = df_other_raw.join(get_data_csv_to_df(f, PATH_TO_OTHER_STOCK_CSV_FILES), on='Date', how='left')


# Calculate diff between days and then the DAYS_FOR_PREDICTION day rolling average
df = get_if_stock_price_increase(df_raw, DAYS_FOR_PREDICTION)
df_other = get_if_stock_price_increase(df_other_raw, DAYS_FOR_PREDICTION)
df_all = df.join(df_other, on='Date', how='left') # Combine all dataframes, df_all is used when predicting


# Test lags from DAYS_ROLLING + 1 to 31 days
results_stock_corr = pd.DataFrame(columns=['stock','corr_stock','lag','corr']) # Define dataframe for results

for lag in range(DAYS_FOR_PREDICTION + 1, 32):
    # Loop over all lag's

    df_lag, df_adjusted_for_lag = calc_lag_and_cut_time_period(df, lag)
    df_other_lag, _ = calc_lag_and_cut_time_period(df_other, lag)
    df_lag = df_lag.join(df_other_lag, on='Date', how='left') # Add df_lag to df_other_lag

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
print(results_stock_corr)

# Build a dataframe for each OMXS30 stock with df_adjusted_for_lag['stock'] and all predictions columns calculated using their lag
# Ex: df_adjusted_for_lag['SEB'], df_lag['BOL']@lag=24, df_lag['BOL']@lag=25, df_lag['SWED']@lag=25
# Test different methods to predict if stock price will increase, Logit, SVM, RF, XGB, NN
# Use results from all predictions to vote 
# Code from: https://stackabuse.com/classification-in-python-with-scikit-learn-and-pandas/
# Models are evaluated using accuracy
# Accuracy is calculated using score. Number of correct predictions / number of predictions

stocks_with_corr = list(set(results_stock_corr['stock'])) # All stocks that has a correlation higher than MIN_CORR_VALUE
results_predictions = pd.DataFrame(columns=['Stock','Logit','SVM','RF','XGB','Vote']) # Define dataframe for results

for stock in stocks_with_corr:
    corr_data_one_stock = results_stock_corr[results_stock_corr['stock'] == stock]
    df_for_prediction = df_adjusted_for_lag[[stock]]

    # Create dataframe with resonse stock in column 0 and then predictor stocks:
    for row in range(0,len(corr_data_one_stock)):
        predictor, _ = calc_lag_and_cut_time_period(df_all[[corr_data_one_stock.iloc[row,1]]], corr_data_one_stock.iloc[row,2])
        predictor.columns = [list(df_all[[corr_data_one_stock.iloc[row,1]]].columns)[0] + '_' + str(corr_data_one_stock.iloc[row,2])]
        df_for_prediction = df_for_prediction.join(predictor, on='Date', how='left')

    # Split in to train and test
    split_row = int(len(df_for_prediction) * 0.66)
    train = df_for_prediction[:split_row]
    test = df_for_prediction[split_row:]
    X_train = train.iloc[:, 1:]
    y_train = train.iloc[:,0]
    X_test = test.iloc[:DAYS_FOR_EVALUATION, 1:] # Only test for the first 7 days in test
    y_test = test.iloc[:DAYS_FOR_EVALUATION,0]

    # Logistic regression
    LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)  
    LR_pred = LR.predict(X_test)  
    LR_res = str(round(LR.score(X_test,y_test), 4))

    # Support vector machine
    SVM = svm.SVC(decision_function_shape='ovo', gamma='scale').fit(X_train, y_train)  
    SVM_pred = SVM.predict(X_test)  
    SVM_res = str(round(SVM.score(X_test, y_test), 4))

    # Random Forrest
    RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(X_train, y_train)  
    RF_pred = RF.predict(X_test)  
    RF_res = str(round(RF.score(X_test, y_test), 4))

    # Extreme boosting machine
    XGB = xgb.XGBClassifier(objective="binary:logistic", random_state=0,)
    XGB.fit(X_train, y_train)
    XGB_pred = XGB.predict(X_test)
    XGB_res = str(round(XGB.score(X_test, y_test), 4))    

    # Neural network
    NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1).fit(X_train, y_train)  
    NN_pred = NN.predict(X_test)  
    NN_res = str(round(NN.score(X_test, y_test), 4))

    # Voting
    prediction_results = pd.DataFrame({'LR':LR_pred, 'SVM':SVM_pred, 'RF':RF_pred, 'XGB':XGB_res, 'NN':NN_pred},index=y_test.index.copy())
    prediction_results['vote'] = prediction_results.mean(axis=1).round()
    Vote_res = str(round(np.where(prediction_results['vote'] == y_test, 1, 0).sum() / len(prediction_results),4))

    # Dataframe with all results
    results_predictions = results_predictions.append(
    {'Stock':stock,
    'Logit':LR_res,
    'SVM':SVM_res,
    'RF':RF_res,
    'XGB':XGB_res,
    'NN':NN_res,
    'Vote':Vote_res},
    ignore_index=True)

    #print(stock+' Logit: '+LR_res+' SVM: '+SVM_res+' RF: '+RF_res+' XGB: '+XGB_res+' NN: '+NN_res+' Vote: '+Vote_res)

results_predictions.sort_values(by='Vote', ascending=False, inplace=True)
print(results_predictions)
