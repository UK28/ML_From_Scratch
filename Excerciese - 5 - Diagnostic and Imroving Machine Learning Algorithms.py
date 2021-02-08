import numpy as np
from scipy.io import loadmat
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data_dict = loadmat('ex5/ex5data1.mat')

# training set
X = data_dict['X']  # change in water level
y = data_dict['y'].ravel()  # flow of water out from the dam

# cross validation set
X_val = data_dict['Xval']
y_val = data_dict['yval'].ravel()

# testing set
X_test = data_dict['Xtest']
y_test = data_dict['ytest'].ravel()

# adding ones column to the training, validation and testing set
X = np.hstack((np.ones((X.shape[0], 1)), X))
X_val = np.hstack((np.ones((X_val.shape[0], 1)), X_val))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# initialising the regularisation parameter
reg_param = 2


def linear_regression_reg_cost_function(theta,
                                        x,
                                        y,
                                        lambda_):
    m = x.shape[0]  # number of training examples

    cost = (1/(2*m))*(np.sum(np.square(np.subtract(np.dot(theta, x.T), y))))
    cost += (lambda_/(2*m))*(np.sum(np.square(theta[1:])))
    return cost


def linear_regression_reg_gradient_function(theta,
                                            x,
                                            y,
                                            lambda_):
    m = x.shape[0]  # number of training examples

    grad = (1/m)*(np.dot(np.subtract(np.dot(theta, x.T), y), x))
    grad[1:] += (lambda_/m)*(theta[1:])
    return grad


def train_linear_regression_function(x, y, lambda_):
    theta_initial = np.ones(x.shape[1])  # initialising the parameter for the first iteration
    result = opt.fmin_cg(f=linear_regression_reg_cost_function,
                         x0=theta_initial,
                         fprime=linear_regression_reg_gradient_function,
                         args=(x, y, lambda_))
    return result


def learning_curve_function(x, y, lambda_=0):  # plots graph of training and validation error w.r.t training size
    error_train = []  # initialising the train error vector. saves training set error
    error_val = []  # initialising the validation error vector. saves validation set error
    for _ in range(1, x.shape[0]+1):
        param = train_linear_regression_function(x[0:_, :], y[0:_], reg_param)  # we include regularisation term while training model
        error_train = np.append(error_train, linear_regression_reg_cost_function(param, x[0:_, :], y[0:_], lambda_))
        error_val = np.append(error_val, linear_regression_reg_cost_function(param, X_val, y_val, lambda_))
    # now plotting the validation and training error w.r.t number of training examples
    plt.plot(np.arange(x.shape[0])+1, error_train, color="Blue", label="Error Train")
    plt.plot(np.arange(x.shape[0])+1, error_val, color="Red", label="Error validation")
    plt.title("Learning Curve")
    plt.xlabel("Number of training examples (m)")
    plt.ylabel("Error")
    plt.grid()
    plt.legend()
    plt.show()


# FITTING THE LINEAR REGRESSION MODEL WITH REGULARISATION AND DIAGNOSING THE MODEL USING LEARNING CURVE
#learning_curve_function(X, y)
"""
DIAGNOSTICS: Examining the learning curve, we found that both training and validation are high which is a clear symptom
of high bias/under-fitting problem which means that our model is too simple to fit the data properly. So, to solve this
problem we have to fit a more complex model like polynomial regression model.
NOTE: Note that increasing lambda_ also won't help solve this problem because it won't have any effect on single feature
"""

# FITTING THE POLYNOMIAL REGRESSION MODEL.


def polynomial_feature_function(x, degree):
    if np.all(x[:, 0] == 1):
        x = np.delete(x, 0, axis=1)
    ones = np.ones((x.shape[0], 1))
    for _ in range(1, degree+1):
        ones = np.hstack((ones, np.power(x, _)))
    return ones


deg = 8
# mapping single feature into polynomial feature
X = polynomial_feature_function(X, deg)
X_val = polynomial_feature_function(X_val, deg)
X_test = polynomial_feature_function(X_test, deg)

# NOTE: converting a single feature into polunomial feature will unscale the data and hence we have to perform normalisation to keep data on same scale
Std_Scalar = StandardScaler()
X[:, 1:] = Std_Scalar.fit_transform(X[:, 1:])
X_val[:, 1:] = Std_Scalar.fit_transform(X_val[:, 1:])
X_test[:, 1:] = Std_Scalar.fit_transform(X_test[:, 1:])

# fitting the polynomial regression model and plotting the learning curve
learning_curve_function(X, y)

"""
DIAGNOSTICS: Learning curve for the polynomial regression model suggests high validation error and low training error.
This huge gap shows the clear symptom of high variance/over-fitting problem which means that our model is fitting well
for the training error but will fail to generalise to the new values which is unknown to the model. 
SOLUTION: One of the best solution for this problem is to include regularisation in the model and increasing lambda.
"""

# NEXT STEP IS TO FIND THE OPTIMUM VALUE OF REGULARISATION PARAMETER FOR THE POLYNOMIAL REGRESSION MODEL
lambda_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
error_train = []
error_val = []
for _ in range(0, len(lambda_list)):
    param = train_linear_regression_function(X, y, lambda_list[_])
    error_train = np.append(error_train, linear_regression_reg_cost_function(param, X, y, 0))
    error_val = np.append(error_val, linear_regression_reg_cost_function(param, X_val, y_val, 0))
plt.plot(lambda_list, error_train, color="Blue", label="Training Error")
plt.plot(lambda_list, error_val, color="Red", label="Validation Error")
plt.title("Error w.r.t lambda")
plt.xlabel("Regularisation Parameter (lambda)")
plt.ylabel("Error")
plt.grid()
plt.legend()
plt.show()

"""
NOTE: More is the over-fitting the more will be the value of regularisation parameter to fit the model properly.
if we are increasing the degree of polynomial than reg param will also be higher and can be easily seen by learning curve
Final Model: Our final model is the polynomial regression model with degree 8 and regularisation parameter as 2
"""
