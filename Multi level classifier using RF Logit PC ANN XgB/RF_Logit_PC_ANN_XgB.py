'''
https://medium.freecodecamp.org/multi-class-classification-with-sci-kit-learn-xgboost-a-case-study-using-brainwave-data-363d7fca5f69

'''
#%%
import pandas as pd
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

#%%
# Load data
brainwave_df = pd.read_csv(os.getcwd() + '/brainwave/data/emotions.csv', index_col=False)
brainwave_df.head()

#%%
# See if the classes have the same size
plt.figure(figsize=(12,5))
sns.countplot(x=brainwave_df.label, color='mediumseagreen')
plt.title('Emotional sentiment class distribution', fontsize=16)
plt.ylabel('Class Counts', fontsize=16)
plt.xlabel('Class Label', fontsize=16)
plt.xticks(rotation='vertical');

#%%
# Save and remove lable form data frame
label_df = brainwave_df['label']
brainwave_df.drop('label', axis = 1, inplace=True)
brainwave_df.head()
#%%
''' Method 
We will use a ‘cross validation’ (in our case will use 10 fold cross validation) approach over 
the dataset and take average accuracy. This will give us a holistic view of the classifier’s accuracy.

We will use a ‘Pipeline’ based approach to combine all pre-processing and main classifier computation. 
A ML ‘Pipeline’ wraps all processing stages in a single unit and act as a ‘classifier’ itself. 
By this, all stages become re-usable and can be put in forming other ‘pipelines’ also.
'''
#%%
''' RandomForest Classifier '''

start = time.time()
pl_random_forest = Pipeline(steps=[('random_forest', RandomForestClassifier(n_estimators=10))])
scores = cross_val_score(pl_random_forest, brainwave_df, label_df, cv=10,scoring='accuracy')

print('Accuracy for RandomForest : ', round(scores.mean()*1000)/1000)
end = time.time()
print('Time for calculation: ', round((end - start)*100)/100, 'seconds')
# Takes about 5 seconds

#%%
''' Logistic Regression Classifier '''

start = time.time()
pl_log_reg = Pipeline(steps=[('scaler',StandardScaler()),
                             ('log_reg', LogisticRegression(multi_class='multinomial', solver='saga', max_iter=400, tol=0.01))])
scores = cross_val_score(pl_log_reg, brainwave_df, label_df, cv=10,scoring='accuracy')

print('Accuracy for Logistic Regression: ', round(scores.mean()*1000)/1000)
end = time.time()
print('Time for calculation: ', round((end - start)*100)/100, 'seconds')
# Takes about 2 minutes

#%%
''' Principal Component Analysis (PCA) '''

start = time.time()
scaler = StandardScaler()
scaled_df = scaler.fit_transform(brainwave_df)

pca = PCA(n_components = 20)
pca_vectors = pca.fit_transform(scaled_df)
for index, var in enumerate(pca.explained_variance_ratio_):
    print("Explained Variance ratio by Principal Component ", (index+1), " : ", round(var*1000)/1000)

end = time.time()
print('Time for calculation: ', round((end - start)*100)/100, 'seconds')

#%%
# Plot the first two principle components
plt.figure(figsize=(25,8))
sns.scatterplot(x=pca_vectors[:, 0], y=pca_vectors[:, 1], hue=label_df)
plt.title('Principal Components vs Class distribution', fontsize=16)
plt.ylabel('Principal Component 2', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=16)
plt.xticks(rotation='vertical');

#%%
'''
In the above plot, three classes are shown in different colours. 
So, if we use the same ‘Logistic Regression’ classifier with these two PCs, 
then from the above plot we can probably say that the first classifier will 
separate out ‘NEUTRAL’ cases from other two cases and the second classifier will 
separate out ‘POSITIVE’ & ‘NEGATIVE’ cases (as there will be two internal logistic 
classifiers for 3-class problem). Let’s try and see the accuracy.
'''

start = time.time()
pl_log_reg_pca_2 = Pipeline(steps=[('scaler',StandardScaler()),
                             ('pca', PCA(n_components = 2)),
                             ('log_reg', LogisticRegression(multi_class='multinomial', solver='saga', max_iter=400, tol=0.001))])
scores = cross_val_score(pl_log_reg_pca_2, brainwave_df, label_df, cv=10,scoring='accuracy')

print('Accuracy for Logistic Regression with 2 Principal Components: ', round(scores.mean()*1000)/1000)
end = time.time()
print('Time for calculation: ', round((end - start)*100)/100, 'seconds')
# Takes about 6 seconds
#%%

# Run with the top ten PC's
start = time.time()
pl_log_reg_pca_10 = Pipeline(steps=[('scaler',StandardScaler()),
                             ('pca', PCA(n_components = 10)),
                             ('log_reg', LogisticRegression(multi_class='multinomial', solver='saga', max_iter=400, tol=0.001))])
scores = cross_val_score(pl_log_reg_pca_10, brainwave_df, label_df, cv=10,scoring='accuracy')

print('Accuracy for Logistic Regression with 10 Principal Components: ', round(scores.mean()*1000)/1000)
end = time.time()
print('Time for calculation: ', round((end - start)*100)/100, 'seconds')
# Takes about 6 seconds

#%%
''' Artificial Neural Network Classifier (ANN) '''

start = time.time()
pl_mlp = Pipeline(steps=[('scaler',StandardScaler()),
                             ('mlp_ann', MLPClassifier(hidden_layer_sizes=(1275, 637)))])
scores = cross_val_score(pl_mlp, brainwave_df, label_df, cv=10,scoring='accuracy')

print('Accuracy for ANN : ', round(scores.mean()*1000)/1000)
end = time.time()
print('Time for calculation: ', round((end - start)*100)/100, 'seconds')
# Takes about 5 minutes

''' Hyper parameters
It is a general convention to start with a hidden layer size of 50 percent of the total data size and subsequent 
layers will be 50 percent of the previous one. In our case these are (1275 = 2549 / 2, 637 = 1275 / 2). 
The number of hidden layers can be taken as hyper-parameter and can be tuned for better accuracy. 
In our case it is 2. 
'''

#%%
''' Extreme Gradient Boosting Classifier (XGBoost) '''

start = time.time()
pl_xgb = Pipeline(steps=[('xgboost', xgb.XGBClassifier(objective='multi:softmax'))])
scores = cross_val_score(pl_xgb, brainwave_df, label_df, cv=10)

print('Accuracy for XGBoost Classifier : ', round(scores.mean()*1000)/1000)
end = time.time()
print('Time for calculation: ', round((end - start)*100)/100, 'seconds')
# Takes about 15 minutes
