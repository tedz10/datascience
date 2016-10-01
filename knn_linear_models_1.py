#
# Ted Zhu
#
# KNN and linear regression problems
#




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




# Model Based Data Imputation

#--------  fill_knn
#input: missing_df (nx2 dataframe, some rows have missing y-values), 
#       full_df (nx2 dataframe, all x-values have correct y-values), 
#       no_y_ind (indices of missing values in missing_df), 
#       with_y_ind (indices of non-missing values in missing_df), 
#       k (integer)
#output: predicted_df (nx2 dataframe, first column is x-vals from missing_df, second column is predicted y-vals), 
#        r (float)


def fill_knn(missing_df, full_df, no_y_ind, with_y_ind, k):
    #preparing data in array form
    
    #training data
    x_train = missing_df.loc[with_y_ind, 'x'].values
    x_train = x_train.reshape((len(with_y_ind), 1)) #make x_train array into a 2D array
    y_train = missing_df.loc[with_y_ind, 'y'].values
    
    #testing data
    x_test = missing_df.loc[no_y_ind, 'x']
    x_test = x_test.values.reshape((len(no_y_ind), 1)) #make x_test array into a 2D array
    y_test = full_df.loc[no_y_ind, 'y'].values
    
    #fit knn model
    neighbours = KNN(n_neighbors=k)
    neighbours.fit(x_train, y_train)
    
    #predict y-values
    predicted_y = neighbours.predict(x_test)
    
    #score predictions
    r = neighbours.score(x_test, y_test)
    
    #fill in missing y-values
    predicted_df = missing_df.copy()
    predicted_df.loc[no_y_ind, 'y'] = pd.Series(predicted_y, index=no_y_ind)
    
    return predicted_df, r


#--------  fill_ling_reg
#input: missing_df (nx2 dataframe, some rows have missing y-values), 
#       full_df (nx2 dataframe, all x-values have correct y-values), 
#       no_y_ind (indices of missing values in missing_df), 
#       with_y_ind (indices of non-missing values in missing_df), 
#       k (integer)
#output: predicted_df (nx2 dataframe, first column is x-vals from missing_df, second column is predicted y-vals), 
#        r (float)


def fill_lin_reg(missing_df, full_df, no_y_ind, with_y_ind):
    #preparing data in array form
    
    #training data
    x_train = missing_df.loc[with_y_ind, 'x'].values
    x_train = x_train.reshape((len(with_y_ind), 1)) #make x_train array into a 2D array
    y_train = missing_df.loc[with_y_ind, 'y'].values
    
    #testing data
    x_test = missing_df.loc[no_y_ind, 'x'].values
    x_test = x_test.reshape((len(no_y_ind), 1)) #make x_test array into a 2D array
    y_test = full_df.loc[no_y_ind, 'y'].values
    
    #fit linear model
    regression = Lin_Reg()
    regression.fit(x_train, y_train)
    
    #predict y-values
    predicted_y = regression.predict(x_test)
    
    #score predictions
    r = regression.score(x_test, y_test)
    
    #fill in missing y-values
    predicted_df = missing_df.copy()
    predicted_df.loc[no_y_ind, 'y'] = pd.Series(predicted_y, index=no_y_ind)
    
    return predicted_df, r




#--------  plot_missing
#input: ax1 (axes), ax2 (axes), 
#       predicted_knn (nx2 dataframe with predicted vals), r_knn (float),
#       predicted_lin (nx2 dataframe with predicted vals), r_lin (float), 
#       k (integer),
#       no_y_ind (indices of rows with missing y-values),
#       with_y_ind (indices of rows with no missing y-values)
#output: ax1 (axes), ax2 (axes)

def plot_missing(ax1, ax2, predicted_knn, r_knn, predicted_lin, r_lin, k, no_y_ind, with_y_ind):
    knn_given = ax1.scatter(predicted_knn.loc[with_y_ind]['x'].values, 
                predicted_knn.loc[with_y_ind]['y'].values, 
                color='blue')

    knn_predict = ax1.scatter(predicted_knn.loc[no_y_ind]['x'].values, 
                predicted_knn.loc[no_y_ind]['y'].values, 
                color='red')

    ax1.set_title('KNN, R^2:' + str(r_knn))
    ax1.set_xlabel('Values of X')
    ax1.set_ylabel('Values of Y')
    ax1.legend((knn_given, knn_predict),
               ("Given", "Predicted"),
               scatterpoints = 1,
               loc='lower right')
    
    #plot given points
    lin_given = ax2.scatter(predicted_lin.loc[with_y_ind]['x'].values, 
                predicted_lin.loc[with_y_ind]['y'].values,
                color='blue')

    #plot points of our model
    lin_predict = ax2.scatter(predicted_lin.loc[no_y_ind]['x'].values, 
                predicted_lin.loc[no_y_ind]['y'].values, 
                color='green')

    ax2.set_title('Lin Reg, R^2:' + str(r_lin))
    ax2.set_xlabel('Values of X')
    ax2.set_ylabel('Values of Y')
    ax2.legend((lin_given, lin_predict),
               ("Given", "Predicted"),
               scatterpoints = 1,
               loc='lower right')
    
    return ax1, ax2




#number of neighbours
k=10

#plot predicted points
#fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 10))


#Read dataset 1
missing_df_1 = pd.read_csv('dataset_2/dataset_1_missing.txt')
full_df_1 = pd.read_csv('dataset_2/dataset_1_full.txt')

missing_df_2 = pd.read_csv('dataset_2/dataset_2_missing.txt')
full_df_2 = pd.read_csv('dataset_2/dataset_2_full.txt')
missing_df_3 = pd.read_csv('dataset_2/dataset_3_missing.txt')
full_df_3 = pd.read_csv('dataset_2/dataset_3_full.txt')
missing_df_4 = pd.read_csv('dataset_2/dataset_4_missing.txt')
full_df_4 = pd.read_csv('dataset_2/dataset_4_full.txt')
missing_df_5 = pd.read_csv('dataset_2/dataset_5_missing.txt')
full_df_5 = pd.read_csv('dataset_2/dataset_5_full.txt')
missing_df_6 = pd.read_csv('dataset_2/dataset_6_missing.txt')
full_df_6 = pd.read_csv('dataset_2/dataset_6_full.txt')
df_lst = [(missing_df_1, full_df_1), (missing_df_2, full_df_2), (missing_df_3, full_df_3), (missing_df_4, full_df_4), (missing_df_5, full_df_5), (missing_df_6, full_df_6)]

#abstrct out as function
def plot_comp (missing_df, full_df):
    
    fig,(ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))

    no_y_ind = missing_df[missing_df['y'].isnull()].index
    with_y_ind = missing_df[missing_df['y'].notnull()].index

    predicted_knn, r_knn = fill_knn(missing_df, 
                                    full_df, 
                                    no_y_ind, 
                                    with_y_ind, 
                                    k)

    predicted_lin, r_lin = fill_lin_reg(missing_df, 
                                        full_df, 
                                        no_y_ind, 
                                        with_y_ind)

    ax1, ax2 = plot_missing(ax1, 
                            ax2, 
                            predicted_knn, r_knn,
                            predicted_lin, r_lin,
                            k,
                            no_y_ind, 
                            with_y_ind)


    plt.show()

    #iterate over list of datasets and plot
for i in xrange(len(df_lst)):
    plot_comp(df_lst[i][0], df_lst[i][1])
    


#Use dataset_1_missing.txt to show impact of the choice of k on the performance of KNN
    
def k_test(missing_df, full_df, k):
    
    #separate to find wich ones are missing/aren't
    no_y_ind = missing_df[missing_df['y'].isnull()].index
    with_y_ind = missing_df[missing_df['y'].notnull()].index
    predicted, r = fill_knn(missing_df, full_df, no_y_ind, with_y_ind, k)
    return r

r_val = []
k_val = []

#iterate over to calculate each r^2 value per k
for i in xrange(1, 345):
    x = k_test(missing_df_1, full_df_1, i)
    r_val.append(x)
    k_val.append(i)
    
#plot k vs r^2 values
plt.scatter(k_val, r_val)
plt.xlabel("# of neighbors")
plt.ylabel("R^2 value")
plt.suptitle("Correlation of # of neighbors used in KNN to R^2 value.")
plt.show()

ind = r_val.index(max(r_val))
print "The k value that returns the highest r^2 value is: " + str(k_val[ind])



#Part 3: Is the Best (Linear Model) Good Enough?

# Use generic fits ex. slope = 0.4, intercept = 0.2, slope = 0.4, intercept = 4
# Then use  linear regression model
# Visualize fit, compute residuals, and residual plots of predicted values along 
# with residuals, as well as a residual histogram.
# Calculate the R^2 coefficients

#REad dataset_1_full.txt
full_1 = pd.read_csv('dataset_2/dataset_1_full.txt')

#Quickly visualize dataset to make first observations
plt.scatter(full_1.iloc[:,0], full_1.iloc[:,1])
plt.xlabel("x values")
plt.ylabel("y values")
plt.suptitle("Dataset_1_full plotted as scatterplot")


def plot_comp(df, slope, constant):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 5))
    
    x_val = df.iloc[:,0]
    y0 = slope * x_val + constant
    
    #print r^2 score
    complete_predict = pd.concat([x_val, y0], axis = 1)
    #sk.metrics.r2_score(df, complete_predict, multioutput = "variance_weighted")
    r2 = score(complete_predict, df)
    print "The r^2 score is: " + str(r2)
    
    #plot all points, plus line
    points_1 = ax1.scatter(x_val, df.iloc[:,1])
    line_fit_1, = ax1.plot(x_val, y0)
    ax1.set_title("Slope:  " + str(slope) + ", Constant: " + str(constant))
    ax1.set_xlabel("x values")
    ax1.set_ylabel("y values")
    ax1.legend((points_1, line_fit_1,),
              ("Actual", "Linear model"),
              scatterpoints = 1,
              loc='best')
    
    #plot residuals, or actual - predicted
    res = df.iloc[:,1] - y0
    points_2 = ax2.scatter(x_val, res, color = "red")
    line_fit_2 = ax2.scatter(x_val, y0)
    ax2.set_title("Plot of residuals and predicted values")
    ax2.set_xlabel("x values")
    ax2.set_ylabel("y values")
    ax2.legend((points_2, line_fit_2,),
              ("Residuals", "Linear model"),
              scatterpoints = 1,
              loc='best')
    
    ax3.hist(res, 50)
    ax3.set_title("Residual histogram")
    ax3.set_xlabel("x values")
    ax3.set_ylabel("y values")
    
    plt.show()
    
plot_comp(full_1, 0.4, 0.2)
plot_comp(full_1, 0.4, 4)
slope, constant = linear_reg_fit(full_1)
plot_comp(full_1, slope, constant)


#read in other datasets
full_2 = pd.read_csv('dataset_2/dataset_2_full.txt')
full_3 = pd.read_csv('dataset_2/dataset_3_full.txt')
full_4 = pd.read_csv('dataset_2/dataset_4_full.txt')
full_5 = pd.read_csv('dataset_2/dataset_5_full.txt')
full_6 = pd.read_csv('dataset_2/dataset_6_full.txt')

#Run fits and plots on them to see effects

slope_2, constant_2 = linear_reg_fit(full_2)
plot_comp(full_2, slope_2, constant_2)

slope_3, constant_3 = linear_reg_fit(full_3)
plot_comp(full_3, slope_3, constant_3)

slope_4, constant_4 = linear_reg_fit(full_4)
plot_comp(full_4, slope_4, constant_4)

slope_5, constant_5 = linear_reg_fit(full_5)
plot_comp(full_5, slope_5, constant_5)

slope_6, constant_6 = linear_reg_fit(full_6)
plot_comp(full_6, slope_6, constant_6)







