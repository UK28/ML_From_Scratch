# Module for loading mat data file in python
from scipy.io import loadmat

# Module for scientific computation in python
import numpy as np

# Module for handling data frames
import pandas as pd

# Module to import optimisation parameter
import scipy.optimize as opt


def co_fi_reg_cost_function(param, Y, R, num_users, num_movies, num_features, lambda_):
    """
    This function computes and returns the squared error and the gradient
    :param param: A single dimensional vector
        Feature matrix X and parameter matrix Theta are being unrolled into single dimensional vector param
    :param Y: A (num_movies, num_users) matrix
        This holds the ratings given by jth user for the ith movie
    :param R: A (num_movies, num_users) matrix
        This holds either the value 1 or 0. 1 means that user has given a rating for the movie while 0 means the user
        hasn't
    :param num_users: A scalar value
    :param num_movies: A scalar value
    :param num_features: A scalar value
    :param lambda_: A floating point value
        Regularisation parameter
    :return: A tuple. First value is a floating point value and the other is a one dimensional vector
        returns squared error and a single dimensional vector which has all the elements of feature and parameter matrix
        unrolled into it
    """
    X = param[:num_movies*num_features].reshape(num_movies, num_features)  # (1682, 10)
    Theta = param[num_movies*num_features:].reshape(num_users, num_features)  # (943, 10)
    J = (1/2)*(np.sum(np.square(np.subtract(np.dot(X, Theta.T)*R, Y))))
    reg_term = (lambda_/2)*(np.sum(np.square(X))+np.sum(np.square(Theta)))
    total_cost = J + reg_term

    Theta_grad = np.zeros_like(Theta)  # (943, 10)
    for j in range(num_users):
        idx = np.where(R[:, j] == 1)[0]  # index of all the movies rated by user j
        Y_temp = Y[idx, [j]].reshape(len(idx), 1)
        Theta_grad[[j], :] = np.dot(np.subtract(np.dot(X[idx, :], Theta[[j], :].T), Y_temp).T, X[idx, :])+(lambda_*Theta[[j], :])

    X_grad = np.zeros_like(X)  # (943, 10)
    for i in range(num_movies):
        idx = np.where(R[i, :] == 1)[0]  # index of all the users rated the movie i
        Y_temp = Y[[i], idx].reshape(1, len(idx))
        X_grad[[i], :] = np.dot(np.subtract(np.dot(X[[i], :], Theta[idx, :].T), Y_temp), Theta[idx, :])+(lambda_*X[[i], :])

    grad = np.append(X_grad.ravel(), Theta_grad.ravel())

    return total_cost, grad


def normalise_ratings_function(Y, R):
    """
    This function takes ratings and the indicator matrix as an input and returns the normalised matrix and mean vector
    :param Y: A (num_movies, num_users) dimensional matrix
        This holds the ratings given by jth user for the ith movie
    :param R: A (num_movies, num_users) dimensional binary indicator matrix
        This holds the value either 1 or 0. 1 means that the user has rated a movie and 0 means not
    :return: A (num_movies, num_users) dimensional matrix and a single dimensional vector
        returns the normalised rating matrix and a single dimensional vector that holds the mean rating of all the movies
        for which the user has rated
    """
    Y_norm = np.zeros_like(Y)
    Y_mean = np.zeros((Y.shape[0], 1))
    for i in range(Y.shape[0]):
        idx = np.where(R[i, :] == 1)[0]  # indices of all the users who have rated a movie i
        Y_mean[i] = np.mean(Y[[i], idx])
        Y_norm[[i], idx] = Y[[i], idx] - Y_mean[i]

    return Y_norm, Y_mean


"""
I am creating a matrix for the list of movies along with creating a my ratings matrix which has ratings of the movies as
per my taste and than I'll be adding this matrix to the original ratings data. At last, I am going to use this ratings
data to train the collaborative filtering model after normalising the data.
"""
movies = pd.read_csv('ex8/movie_ids.txt', encoding='latin-1', names=['Movies']).to_numpy()  # list all movies

'''creating my own ratings and adding it to the original ratings data'''
my_ratings = np.zeros((len(movies), 1))

# For example, Toy Story (1995) has ID 1, so to rate it "4", set
my_ratings[0] = 4

# Or suppose did not enjoy Silence of the Lambs (1991),set
my_ratings[97] = 2

# selected a few movies liked / did not like
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

for i in range(my_ratings.shape[0]):
    if my_ratings[i] > 0:
        print("Rated", int(my_ratings[i]), " for", str(movies[i]))

# Loading the ratings and indicator matrix data from mat file
ratings_data = loadmat('ex8/ex8_movies.mat')
Y = ratings_data['Y']
R = ratings_data['R']

# Adding my_ratings to the original ratings data
Y = np.hstack((my_ratings, Y))
R = np.hstack((my_ratings != 0, R))

# Some useful information
num_movies, num_users = Y.shape
num_features = 10

# Normalising the data
Y_norm = normalise_ratings_function(Y, R)[0]
Y_mean = normalise_ratings_function(Y, R)[1]

# Randomly initialising the feature and parameter matrix
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

# Rolling all the parameters into a single dimensional vector
params = np.append(X.ravel(), Theta.ravel())

# Initialising the regularisation parameter
reg_param = 10


# This is the time to put all pieces together into an optimisation function like fmin_cg
def reduced_cost_function(p):
    return co_fi_reg_cost_function(p, Y_norm, R, num_users, num_movies, num_features, reg_param)


opt_parameters = opt.minimize(fun=reduced_cost_function,
                              x0=params,
                              method='CG',
                              jac=True,
                              options={'maxiter': 100})

final_params = opt_parameters.x

# rolling the feature and parameter matrices X and Theta out from the final_params
X = final_params[:num_movies*num_features].reshape(num_movies, num_features)
Theta = final_params[num_movies*num_features:].reshape(num_users, num_features)

prediction_matrix = np.dot(X, Theta.T)  # (1682, 944)
my_prediction = prediction_matrix[:, [0]] + Y_mean

sorted_prediction = np.sort(my_prediction)  # sorting the predicted ratings in ascending order
sorted_idx = sorted_prediction.ravel().argsort()  # getting the indices of the sorted predicted ratings

print("Top recommendations for you")
for i in range(1, 10):
    j = sorted_idx[-i]
    print(i, "Predicted rating", float(my_prediction[j]), ":", str(movies[j]))