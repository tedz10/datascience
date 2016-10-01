# Multiple linear regression
# Implement multiple linear regression from scratch

### Functions for fitting and evaluating multiple linear regression
### Functions for fitting and evaluating multiple linear regression
​
#--------  multiple_linear_regression_fit
# A function for fitting a multiple linear regression
# Fitted model: f(x) = x.w + c
# Input: 
#      x_train (n x d array of predictors in training data)
#      y_train (n x 1 array of response variable vals in training data)
# Return: 
#      w (d x 1 array of coefficients) 
#      c (float representing intercept)
​
def multiple_linear_regression_fit(x_train, y_train):
    
    # Append a column of one's to x
    n = x_train.shape[0]
    ones_col = np.ones((n, 1))
    
    x_train = np.concatenate((x_train, ones_col), axis=1)
    
    # Compute transpose of x
    x_transpose = np.transpose(x_train)
    
    # Compute coefficients: w = inv(x^T * x) x^T * y
    # Compute intermediate term: inv(x^T * x)
    # Note: We have to take pseudo-inverse (pinv), just in case x^T * x is not invertible 
    x_t_x_inv = np.linalg.pinv(np.dot(x_transpose, x_train))
    
    # Compute w: inter_term * x^T * y 
    w = np.dot(np.dot(x_t_x_inv, x_transpose), y_train)
    
    # Obtain intercept: 'c' (last index)
    c = w[-1]
    
    return w[:-1], c
​
#--------  multiple_linear_regression_score
# A function for evaluating R^2 score and MSE 
# of the linear regression model on a data set
# Input: 
#      w (d x 1 array of coefficients)
#      c (float representing intercept)
#      x_test (n x d array of predictors in testing data)
#      y_test (n x 1 array of response variable vals in testing data)
# Return: 
#      r_squared (float) 
#      y_pred (n x 1 array of predicted y-vals)
​
def multiple_linear_regression_score(w, c, x_test, y_test):        
    # Compute predicted labels
    y_pred = np.dot(x_test, w) + c
    
    # Evaluate sqaured error, against target labels
    # sq_error = \sum_i (y[i] - y_pred[i])^2
    sq_error = np.sum(np.square(y_test - y_pred))
    
    # Evaluate squared error for a predicting the mean value, against target labels
    # variance = \sum_i (y[i] - y_mean)^2
    y_mean = np.mean(y_test)
    y_variance = np.sum(np.square(y_test - y_mean))
    
    # Evaluate R^2 score value
    r_squared = 1 - sq_error / y_variance
​
    return r_squared, y_pred

# Load train and test data sets
data_train = np.loadtxt('datasets/dataset_1_train.txt', delimiter=',', skiprows=1)
data_test = np.loadtxt('datasets/dataset_1_test.txt', delimiter=',', skiprows=1)
​
# Split predictors from response
# Training
y_train = data_train[:, -1]
x_train = data_train[:, :-1]
​
# Testing
y_test = data_test[:, -1]
x_test = data_test[:, :-1]
​
# Fit multiple linear regression model
w, c = multiple_linear_regression_fit(x_train, y_train)
​
# Evaluate model
r_squared, _ = multiple_linear_regression_score(w, c, x_test, y_test)
​
print 'R^2 score on test set:', r_squared
R^2 score on test set: 0.177944627327
Part (b): Confidence interval on regression parameters
Using your linear regression implementation from Part (a), model the data in dataset_2.txt, which contains five predictor variables in the first five columns, and the response variable in the last column.
Compute confidence intervals for the model parameters you obtain:
Create 200 random subsamples of the data set of size 100, and use your function to fit a multiple linear regression model to each subsample.
For each coefficient on the predictor variables: plot a histogram of the values obtained across the subsamples, and calculate the confidence interval for the coefficients at a confidence level of 95%.
Highlight the mean coeffcient values and the end points of the confidence intervals using vertical lines on the histogram plot. How large is the spread of the coefficient values in the histograms, and how tight are the confidence intervals?
Use the formula for computing confidence intervals provided in class (or use statmodels) to compute the the confidence intervals. Compare confidence intervals you find through simulation to the ones given by the formula (or statmodels), are your results what you would expect?
Note: You may not use pre-built models or model evaluators for these tasks.
In [4]:

data = np.loadtxt("datasets/dataset_2.txt", delimiter=',', skiprows = 1)
​
# Size of data set, and subsample (10%)
x = data[:, :-1]
y = data[:, -1]
​
# Record size of the data set
n = x.shape[0]
d = x.shape[1]
subsample_size = 100
​
# No. of subsamples
num_samples = 200
    
### Linear regression with all 5 predictors
​
# Create a n x d array to store coefficients for 100 subsamples
coefs_multiple = np.zeros((num_samples, d))
​
print 'Linear regression with all predictors'
​
# Repeat for 200 subsamples
for i in range(num_samples):
    # Generate a random subsample of 50 data points
    perm = np.random.permutation(n) # Generate a list of indices 0 to n and permute it
    x_subsample = x[perm[:subsample_size], :] # Get x-vals for the first 50 indices in permuted list
    
    y_subsample = y[perm[:subsample_size]] # Get y-vals for the first 50 indices in permuted list
​
    # Fit linear regression model on subsample
    w, c = multiple_linear_regression_fit(x_subsample, y_subsample)
    # Store the coefficient for the model we obtain
    coefs_multiple[i, :] = w
​
# Plot histogram of coefficients, and report their confidence intervals 
fig, axes = plt.subplots(1, d, figsize=(20, 3))
​
# Repeat for each coefficient
for j in range(d):
    # Compute mean for the j-th coefficent from subsamples
    coef_j_mean = np.mean(coefs_multiple[:, j])
    
    # Compute confidence interval at 95% confidence level (use formula!)
    conf_int_left = np.percentile(coefs_multiple[:, j], 2.5)
    conf_int_right = np.percentile(coefs_multiple[:, j], 97.5)
       
    # Plot histogram of coefficient values
    axes[j].hist(coefs_multiple[:, j], alpha=0.5)
​
    # Plot vertical lines at mean and left, right extremes of confidence interval
    axes[j].axvline(x = coef_j_mean, linewidth=3)
    axes[j].axvline(x = conf_int_left, linewidth=1, c='r')
    axes[j].axvline(x = conf_int_right, linewidth=1, c='r')
    
    # Set plot labels
    axes[j].set_title('[' + str(round(conf_int_left, 4)) 
                      + ', ' 
                      + str(round(conf_int_right, 4)) + ']')
    axes[j].set_xlabel('Predictor ' + str(j + 1))
    axes[j].set_ylabel('Frequency')
    
    print "The spread is represented by the standard deviation: " + str(np.std(coefs_multiple[:, j])) + " for graph " + str(j + 1)
​
plt.show()
Linear regression with all predictors
The spread is represented by the standard deviation: 0.162379982657 for graph 1
The spread is represented by the standard deviation: 0.326446833095 for graph 2
The spread is represented by the standard deviation: 0.316250002223 for graph 3
The spread is represented by the standard deviation: 0.237090193606 for graph 4
The spread is represented by the standard deviation: 0.336824218308 for graph 5


# Add column of ones to x matrix
x = sm.add_constant(x)
​
# Create model for linear regression
model = sm.OLS(y, x)
# Fit model
fitted_model = model.fit()
# The confidence intervals for our five coefficients are contained in the last five
# rows of the fitted_model.conf_int() array
conf_int = fitted_model.conf_int()[1:, :]
​
for j in range(d):
    print 'The confidence interval for the', j + 1, 'th coefficient: [', conf_int[j][0], ',', conf_int[j][1], ']'
The confidence interval for the 1 th coefficient: [ 0.552772624515 , 0.75038508161 ]
The confidence interval for the 2 th coefficient: [ 0.352230409656 , 0.749743369435 ]
The confidence interval for the 3 th coefficient: [ 0.0889138463555 , 0.47338926001 ]
The confidence interval for the 4 th coefficient: [ 0.809809940402 , 1.09854837094 ]
The confidence interval for the 5 th coefficient: [ 0.0785426153804 , 0.488433606409 ]
The confidence intervals for the simulations we generated are much larger in range than the computed ones for all five instances. We can expect this because the actual dataset (dataset2) that is used for the computations, and since we have more datapoints (1000 datapoints vs 100 datapoints in the simulations) we can be more confident. This tightens the confidence interval considerably.




#Implement polynomial regression from scratch

#takes as input: training set, x_train, y_train and the degree of the polynomial
#fits a polynomial regression model
#returns the model parameters (array of coefficients and the intercept)
​
data_3 = pd.read_csv("datasets/dataset_3.txt")
​
def polynomial_regression_fit(x_train, y_train, degree):
    
    #get the data size
    n = np.size(y_train)
    x_poly_d = np.zeros([n, degree])
    
    #scale up by powers
    for d in xrange(1, degree + 1):
        x_poly_d[:, d - 1] = np.power(x_train, d)
    
    #fit using linear regression since it is an extension
    w = multiple_linear_regression_fit(x_poly_d, y_train)
    
    return w
​
#split into separate columns
x_train = data_3['x']
y_train = data_3[' y']
​
#test to make sure function is correct
w = polynomial_regression_fit(x_train, y_train, 3) 

#takes as input: the model parameters 
#(array of coefficients and the intercept), the degree of the polynomial 
#and the test set predictors x_test
#returns the response values predicted by the model on the test set.
def polynomial_regression_predict (params, degree, x_test):
    n = x_test.shape[0]
    x_poly = np.zeros([n, degree])
    for d in xrange(1, degree + 1):
        x_poly[:, d - 1] = np.power(x_test, d)
    
    #concat a column of ones to x matrix
    ones_col = np.ones((n, 1))
    Xt = np.concatenate((ones_col, x_poly), axis=1)
    
    #make into one
    betas = np.insert(params[0], 0, params[1])
​
    #reshape to fit
    reshaped = np.reshape(betas, (-1, degree + 1))
​
    #take dot product
    y_pred = np.dot(reshaped, Xt.T)
    
    return y_pred
​
#test to make sure function works
predicted = polynomial_regression_predict(w, 3, x_train)

#takes an array of predicted response values and 
#the array of true response values y_test
#returns R^2 score for the model on the test set, 
#as well as the sum of squared errors
def polynomial_regression_score(predicted, y_test):
    
    # evaluate sqaured error, against target labels
    sq_error = np.sum(np.square(y_test - predicted))
    
    # evaluate squared error for a predicting the mean value, against target labels
    y_mean = np.mean(y_test)
    y_variance = np.sum(np.square(y_test - y_mean))
    
    # evaluate R^2 score value
    r_squared = 1 - (sq_error / y_variance)
    
    return r_squared, sq_error
​
#test to make sure function works
r, sq_er = polynomial_regression_score(predicted, np.array(y_train))

#split between min/max of x_train
points = np.linspace(x_train.min(), x_train.max(), num = 50)
​
#run fit and predict on each set
fit_3 = polynomial_regression_fit(x_train, y_train, 3)
predict_3 = polynomial_regression_predict(fit_3, 3, points).ravel()
​
fit_5 = polynomial_regression_fit(x_train, y_train, 5)
predict_5 = polynomial_regression_predict(fit_5, 5, points).ravel()
​
fit_10 = polynomial_regression_fit(x_train, y_train, 10)
predict_10 = polynomial_regression_predict(fit_10, 10, points).ravel()
​
fit_25 = polynomial_regression_fit(x_train, y_train, 25)
predict_25 = polynomial_regression_predict(fit_25, 25, points).ravel()
​
​
#set size
plt.figure(figsize=(20,12))
​
#plot
known = plt.scatter(x_train, y_train, color='blue')
plot_3, = plt.plot(points, predict_3, color='red')
plot_5, = plt.plot(points, predict_5, color='green')
plot_10, = plt.plot(points, predict_10, color='yellow')
plot_25, = plt.plot(points, predict_25, color='teal')
​
#add details
plt.legend((known, plot_3, plot_5, plot_10, plot_25),("Known", "Degree 3", "Degree 5", "Degree 10", "Degree 25"))
plt.xlabel('X values', fontsize = 16)
plt.ylabel('Y values', fontsize = 16)
plt.suptitle("Predicted values with degrees 3/5/10/25", fontsize=22)
plt.show()


def plot_poly(d, x_train, y_train):
    
    w = polynomial_regression_fit(x_train, y_train, d)
    predicted = polynomial_regression_predict(w, d, x_train)
    
    #plot fit against actual points
    actual = plt.scatter(x_train, y_train, color = "green")
    predicted = plt.scatter(x_train, predicted, color = "red")
    
    #add details
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.suptitle("Polynomial fit with " + str(d))
    plt.legend((actual, predicted), ('Actual', 'Predicted'))
    plt.show()
    
degrees = [3, 5, 10, 25]
​
#visualize each individual graph
for i in degrees:
    plot_poly(i, x_train, y_train)




#Comparing training and test errors


#split first half into training
rows = data_3.shape
​
mid = rows[0] / 2
​
#split train data
train = data_3.ix[0:(mid - 1), :]
train_x = train['x']
train_y = train[' y']
​
#split test data
test = data_3.ix[mid:rows[0], :]
test_x = test['x']
test_y = test[' y']
​

def plot_poly_1 (d, x_train, y_train, x_test, y_test):
    
    #create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))
    
    #fit and predict the data using training data
    w_train = polynomial_regression_fit(x_train, y_train, d)
    predicted_train = polynomial_regression_predict(w_train, d, x_train)
    r_tr, _ = polynomial_regression_score(predicted_train, np.array(y_train))
    
    #plot
    train_actual = ax1.scatter(x_train, y_train)
    train_fit = ax1.scatter(x_train, predicted_train, color = "red")
    ax1.set_title("Train data + fit with degree: " + str(d) + " and R^2 = " + str(r_tr))
    ax1.set_xlabel("x values")
    ax1.set_ylabel("y values")
    ax1.legend((train_actual, train_fit,),
              ("Actual", "Prediction"),
              scatterpoints = 1,
              loc='best')
    
    #fit on train data and test on test data
    w_test = polynomial_regression_fit(x_train, y_train, d)
    predicted_test = polynomial_regression_predict(w_test, d, x_test)
    r_test, _ = polynomial_regression_score(predicted_test, np.array(y_test))
    
    #plot
    test_actual = ax2.scatter(x_test, y_test)
    test_fit = ax2.scatter(x_test, predicted_test, color = "red")
    ax2.set_title("Test data + fit with degree: " + str(d) + " and R^2 = " + str(r_test))
    ax2.set_xlabel("x values")
    ax2.set_ylabel("y values")
    ax2.legend((test_actual, test_fit,),
              ("Actual", "Prediction"),
              scatterpoints = 1,
              loc='best')
    return r_tr, r_test
​
r_tr_lst = []
r_test_lst = []
degrees = []
​
#loop through 1 - 15 and plot to see actual vs predicted points
for i in xrange(1, 16):    
    r_tr, r_test = plot_poly_1 (i, train_x, train_y, test_x, test_y)
    r_tr_lst.append(r_tr)
    r_test_lst.append(r_test)
    degrees.append(i)
    



#plot r^2 values for train and test data for each degree
tr, = plt.plot(degrees, r_tr_lst, color = "red")
tst, = plt.plot(degrees, r_test_lst, color = "green")
plt.xlabel("Degrees")
plt.ylabel("R^2 value")
plt.legend((tr, tst), ("R^2 for train", "R^2 for test"), loc = 4)
plt.suptitle("R^2 values for train and test fits")
plt.show()



def best_model(degree, train):
    
    #split train data into respective x and y columns
    train_x = np.array(train.iloc[:, 0])
    train_y = np.array(train.iloc[:, 1])
    size = train.shape[0]
    
    #fit onto train data, then predict on same train data
    w_train = polynomial_regression_fit(train_x, train_y, degree)
    predicted_train = polynomial_regression_predict(w_train, degree, train_x)
    
    #calculate rss value
    rss = np.sum((train_y - predicted_train) ** 2)
    
    #calculate aic and bic
    aic = size * np.log((rss / size)) + 2 * degree
    bic = size * np.log((rss / size)) + degree * np.log(size)
    
    return aic, bic
​
best_model(3, train)
​
aic_lst = []
bic_lst = []
degrees = []
​
for i in xrange(1, 16):
    x = best_model(i, train)
    degrees.append(i)
    aic_lst.append(x[0])
    bic_lst.append(x[1])
    
#visualize aic add bic values for each degree value
aic, = plt.plot(degrees, aic_lst, color = "green")
bic, = plt.plot(degrees, bic_lst, color = "red")
plt.xlabel("Degrees of polynomial")
plt.ylabel("AIC/BIC values")
plt.suptitle("AIC and BIC values for different degree polynomials")
plt.legend((aic, bic), ("AIC value", "BIC value"))
plt.show()
​
#select index for lowest aic and bic value
aic_index = aic_lst.index(min(aic_lst)) + 1
bic_index = bic_lst.index(min(bic_lst)) + 1
​
print "The model chosen by the AIC values is: " + str(aic_index) + "."
print "\n"
print "The model chosen by the BIC values is: " + str(bic_index) + "."


#read in data
#data = pd.read_csv("datasets/green_tripdata_2015-01.csv", skiprows = [0])
data = pd.read_csv('https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2015-01.csv', sep = ",")

#separate pick up times into own column
times = np.array(data.iloc[:,1])
passengers = np.array(data.iloc[:,9])
pickuptimes = []
In [72]:

#quick function to convert hours to minutes
def hr_to_min(time):
    cleaned = time.split(":")
    int_lst = []
    for i in cleaned:
        int_lst.append(int(i))
    minutes = int_lst[0] * 60 + int_lst[1]
    return minutes
In [73]:

#loop through and convert to minutes, then add to list
for time in times:
    pickuptime = hr_to_min(time[11:16])
    pickuptimes.append(pickuptime)
In [74]:

#split so that 70% of data is now training data
pickuptimes_train = pickuptimes[:int(.7 * len(pickuptimes))]
​
#use itemfreq function to total how many times each value shows up
frequency = sp.stats.itemfreq(pickuptimes_train)

#convert to df because of what previous functions accept
freq_df = pd.DataFrame(frequency)
​
aic_lst_1 = []
bic_lst_1 = []
degrees_1 = []
​
#calculate aic and bic for degree values
for i in xrange(1, 100):
    x = best_model(i, freq_df)
    degrees_1.append(i)
    aic_lst_1.append(x[0])
    bic_lst_1.append(x[1])
    
aic_best = aic_lst_1.index(min(aic_lst_1)) + 1
bic_best = bic_lst_1.index(min(bic_lst_1)) + 1
print "Best model for AIC is: " + str(aic_best), "Best model for BIC is: " + str(bic_best)
​
#visualize best aic and bic values
plt.scatter(degrees_1, aic_lst_1, color = "red")
plt.scatter(degrees_1, bic_lst_1, color = "green")
plt.xlabel("Degrees")
plt.ylabel("AIC/BIC values")
plt.suptitle("AIC and BIC values for cab data")
plt.show()
​
print "Best model for AIC is: 47; Best model for BIC is: 32"



# pickuptimes_train = pickuptimes[:int(.7 * len(pickuptimes))]
# passengers_train = passengers[:int(.7 * len(passengers))]
​
# rides_arr = np.column_stack((pickuptimes_train, passengers_train))
# rides_df = pd.DataFrame(rides_arr)
​
# # pickuptimes_train_fit = polynomial_regression_fit(pickuptimes_train, passengers_train, degree)
# # predicted_test = polynomial_regression_predict(pickuptimes_train_fit, degree, x_test)
​
# #best_model(3, rides_df)
​
#separate into times and number of pickups
freq_time = freq_df.iloc[:, 0]
freq_val = freq_df.iloc[:, 1]
In [100]:

#quick visualization using previous functions to better look at data
plot_poly(32, freq_time, freq_val)
plt.show()
​
plot_poly(47, freq_time, freq_val)
plt.show()


def density_predict(time):
    minutes = hr_to_min(time)
    #takes: x_train, y_train, degree
    fit_32 = polynomial_regression_fit(freq_time, freq_val, 32)
    #takes: params, degree, x_test
    density_32 = polynomial_regression_predict(fit_32, 32, pd.Series(minutes))
    
    fit_47 = polynomial_regression_fit(freq_time, freq_val, 47)
    density_47 = polynomial_regression_predict(fit_47, 47, pd.Series(minutes)) 
    
    return float('%.02f' %density_32), float('%.02f' %density_47)
​
time = '3:00'
x = density_predict(time)
print x
​
print "Density of cabs at time " + str(time) + " is " + str(x[0]) + " using degree of 32 and " + str(x[1]) + " using degree of 47."
(466.85, 466.48)
Density of cabs at time 3:00 is 466.85 using degree of 32 and 466.48 using degree of 47.