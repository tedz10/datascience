#import necessary libraries

import numpy as np
import pandas as pd
import random
import sklearn as sk
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.cross_validation import train_test_split as sk_split
from sklearn.linear_model import LinearRegression as Lin_Reg
import time
import matplotlib
import matplotlib.pyplot as plt
import time as time
#%matplotlib inline


##################################
#                                #
#  Implement the models by hand  #
#                                #
##################################

# Write functions from scratch
# Function: split (inputs: nx2 dataframe, float m)
# return: nx2 dataframe train & nx2 dataframe test, with m percent and (100 - m) percent

#splits dataset into train and test data
def split(data, m):
    
    #multiply to get decimal
    decimal = m * 0.01
    train = data.sample(frac= decimal, replace=False)
    test = data.drop(train.index)
    
    return train, test

# Implement K-Nearest Neighbour for predicting quantitative variable.
# input: integer k, n x 2 dataframe training set train, n x 1 dataframe testing set test
# return: nx2 dataframe (first column is test and second column is the predicted values)

def knn_predict(k, train, test):
    
    #create column to add predictions in
    predictions = []
    
    for index, test_row in test.iterrows():
        distances = []
        pred = []
        
        #iterate through and store tuples
        for index, train_row in train.iterrows():
            dis = abs(train_row[0] - test_row[0])
            distances.append(dis)
            pred.append(train_row[1])
        
        #sort
        tup_lst = pd.DataFrame({'distance' : distances, 'y' : pred})
        tup_lst.sort_values('distance', inplace = True)
        
        #take first k values
        neighbors = tup_lst.iloc[0:k, :]

        prediction_means = np.mean(neighbors['y'])
        predictions.append(prediction_means)
            
    #add predictions to df
    test['predictions'] = predictions
    
    return test

# Implement linear regression for predicting a quantitative variable
# input: an nx2 dataframe training set train
# return: coefficients of linear regression model - a float slope and a float intercept

def linear_reg_fit (train):
    
    #create and make separate matrixes
    train_x = train.as_matrix(['x'])
    train_y = train.as_matrix(['y'])
    
    #calculate slope/constant
    slope = np.sum((train_x - np.mean(train_x)) * (train_y - np.mean(train_y))) / np.sum((train_x - np.mean(train_x)) ** 2)
    constant = np.mean(train_y) - slope * np.mean(train_x)
    
    return slope, constant

# Write a function linear_reg_predict that satisfies:
# input: nx1 dataframe testing set test, coefficients of the linear regression model
# return: nx2 dataframe (first column is 'test' and second column is predicted values)

def linear_reg_fit (train):
    
    #create and make separate matrixes
    train_x = train.as_matrix(['x'])
    train_y = train.as_matrix(['y'])
    
    #calculate slope/constant
    slope = np.sum((train_x - np.mean(train_x)) * (train_y - np.mean(train_y))) / np.sum((train_x - np.mean(train_x)) ** 2)
    constant = np.mean(train_y) - slope * np.mean(train_x)
    
    return slope, constant

# Implement a function score that satisfies:
# input: an nx2 dataframe predicted, an nx2 dataframe actual
# return: R^2 coefficient of the fit of the predicted values.

def score(predicted, actual):
    
    #calculate residual sum of squares
    rss = np.sum((actual.iloc[:,1] - predicted.iloc[:,1]) ** 2)
    
    #calculate total sum of squares
    tss = np.sum((actual.iloc[:,1] - np.mean(actual.iloc[:,1])) ** 2)
    
    r2 = 1 - (rss/tss)
    
    return r2



#########################################
#                                       #
#  Load contents of dataset_1_full.txt  #
#                                       #
#########################################

df_1_full = pd.read_csv('dataset_2/dataset_1_full.txt')


# Use previous functions to split the data

x, y = split(df_1_full, 70)
testing = pd.DataFrame(y.ix[:,0])

# Evaluate how KNN/linear regression each perform

#knn
knn = knn_predict(50, x, pd.DataFrame(y.ix[:,0]))

print "The r^2 value of our knn model: " + str(score(knn, y))

#linear reg
slope, constant = linear_reg_fit(x)
lin = linear_reg_predict(pd.DataFrame(y.ix[:,0]), slope, constant)
score(lin, y)

print "The r^2 value of our regression model: " + str(score(lin, y))


# Use sklearn to split the data into training and testing sets (70-30). 
# Use sklearn to evaluate how KNN/linear regression each perform on this dataset.


train, test = sk_split(df_1_full, train_size = 0.7)

k = 4

#create separate train/test columns
train_x = train.as_matrix(['x'])
train_y = train.as_matrix(['y'])
test_x = test.as_matrix(['x'])
test_y = test.as_matrix(['y'])

#sklearn split as per documentation example
train, test = sk_split(df_1_full, train_size = 0.7)

#sklearn KNN
neighbors = KNN(n_neighbors = k)
neighbors.fit(train_x, train_y)
predicted_y = neighbors.predict(test_x)
r = neighbors.score(test_x, test_y)

print "The r^2 value of this knn model: " + str(r)


#sklearn linear regression as per documentation example
regression = Lin_Reg()
regression.fit(train_x, train_y)
predicted_y = regression.predict(test_x)
r = regression.score(test_x, test_y)

print "The r^2 value of the regression model: " + str(r)





















