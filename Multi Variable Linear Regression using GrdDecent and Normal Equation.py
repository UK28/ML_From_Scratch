import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import time

# SETTING UP A CONNECTION ENGINE WITH SQLALCHEMY


def setConnection():
    engine = create_engine('mysql+pymysql://root:12345@localhost:3306/testdb')
    connection = engine.connect()
    return connection


def fetchData(query):
    data_frame = pd.read_sql(query, con=setConnection())
    return data_frame


query = 'SELECT * FROM HOUSING_DATA'
housing_data = fetchData(query)

# ADDING A ONES COLUMN TO THE DATA FRAME
#housing_data.insert(loc=0, value=1, column='Ones')

# CREATING FEATURE AND OBSERVATION MATRIX
#feature = np.array(housing_data.loc[:, ["Ones", "Area", "No. of Bedrooms"]])
#observation = np.array(housing_data.loc[:, ["Profit"]])

# DEFINING A FUNCTION CALLED NORMALISE WHICH WILL NORMALISE A MATRIX PASSED TO IT


def norm(matrix):
    no_of_features = len(matrix.T)
    no_of_observation = len(matrix)
    mean = np.zeros((1, no_of_features-1))
    std_dev = np.zeros((1, no_of_features-1))
    for i in range(1, no_of_features):
        mean[:, i-1] = np.mean(matrix[:, i])
        std_dev[:, i-1] = np.std(matrix[:, i])
    matrix = np.divide(np.subtract(matrix[:, 1:no_of_features], mean), std_dev)
    matrix = np.insert(matrix, 0, 1, axis=1)
    return matrix


'''GRADIENT DECENT STARTS FROM HERE'''

def computeCost(theta, feature, observation):
    m = len(feature)
    cost = (1/2*m)*(np.sum(np.square(np.subtract(np.dot(feature, theta), observation))))
    return cost


def gradientDecent(theta, feature, observation, alpha=0.1, iteration=1500):
    m = len(feature)
    theta_history = np.zeros((iteration, len(feature.T)))
    cost_history = np.zeros((iteration, len(feature.T)))
    for i in range(0, iteration):
        theta -= (alpha/m)*(np.dot(feature.T, np.subtract(np.dot(feature, theta), observation)))
        theta_history[i, :] = theta.T
        cost_history[i] = computeCost(theta, feature, observation)
    return theta, theta_history, cost_history


def gdMain(data_frame, alpha, iteration):
    data_frame.insert(loc=0, value=1, column='Ones')
    feature = np.array(housing_data.loc[:, ["Ones", "Area", "No. of Bedrooms"]])# THIS LINE HAS DEPENDENCY ON COLUMN NAMES
    observation = np.array(housing_data.loc[:, ["Profit"]])  # THIS LINE TOO HAS DEPENDENCY ON COLUMN NAME
    theta = np.zeros((len(feature.T), 1))
    theta_GD, theta_history, cost_history = gradientDecent(theta, norm(feature), observation,\
                                                                                alpha=alpha, iteration=iteration)
    return theta_GD, theta_history, cost_history


time1 = time.time()
print(gdMain(housing_data, alpha=0.01, iteration=2500)[0])
time2 = time.time()
print("TIME TAKEN BY GRADIENT DECENT TO RUN(THIS INCLUDES TIME TAKEN TO NORMALISE MATRIX): ", time2-time1)
'''GRADIENT DECENT ENDS HERE'''


def normalEquation(feature, observation):
    theta = np.dot(np.dot(np.linalg.inv(np.dot(feature.T, feature)), feature.T), observation)
    cost = computeCost(theta, feature, observation)
    return theta, cost


def NEmain(data_frame):
    # data_frame.insert(loc=0, value=1, column='Ones')
    feature = np.array(housing_data.loc[:, ["Ones", "Area", "No. of Bedrooms"]])
    observation = np.array(housing_data.loc[:, ["Profit"]])
    theta_NE, cost = normalEquation(norm(feature), observation)
    return theta_NE, cost


theta_NE, cost = NEmain(housing_data)

time3 = time.time_ns()
print(NEmain(housing_data))
time4 = time.time_ns()
print("TIME TAKEN BY NORMAL EQUATION TO RUN(THIS INCLUDES TIME TAKEN TO NORMALISE MATRIX): ", time4-time3)