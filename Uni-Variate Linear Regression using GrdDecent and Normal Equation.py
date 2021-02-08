''' UNI-VARIATE LINEAR REGRESSION USING GRADIENT DECENT AND NORMAL EQUATION ALGORITHM ON FOOD TRUCK DATA'''

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import time
import matplotlib.pyplot as plt


# THIS PART IS ONLY FOR FETCHING DATA FROM DATABASE
def setConnection():
    engine = create_engine('mysql+pymysql://root:12345@localhost:3306/testdb')
    connection = engine.connect()
    return connection


def fetchData(query):
    data_frame = pd.read_sql(query, con=setConnection())
    return data_frame

food_truck_data = fetchData('SELECT * FROM FOOD_TRUCK')

'''USING GRADIENT DECENT AS OUR ALGORITHM'''
time1 = time.time()

# ADDING ONES COLUMN TO OUR FOOD TRUCK DATA
food_truck_data.insert(loc=0, column='Ones', value=1)

# CREATING A MATRIX FOR FEATURE AND OBSERVATION
feature = np.array(food_truck_data.loc[:, ["Ones", "Population"]])
observation = np.array(food_truck_data.loc[:, ["Profit"]])


# IMPLEMENTING COST FUNCTION
def computeCost(theta, feature, obesrvation):
    m = len(feature)
    cost = (1/2*m)*np.sum(np.square(np.subtract(np.dot(feature, theta), observation)))
    return cost


# IMPLEMENTING GRADIENT DECENT FUNCTION
def gradientDecent(theta, feature, observation, alpha=0.01, iteration=3000):
    m = len(feature)
    cost_history = np.zeros((iteration, 1))
    theta_history = np.zeros((iteration, 2))
    for _ in range(0, iteration):
        theta -= (alpha/m)*(np.dot(feature.T, np.subtract(np.dot(feature, theta), observation)))
        theta_history[_, :] = theta.T
        cost_history[_] = computeCost(theta, feature, observation)
    return theta, theta_history, cost_history


# IMPLEMENTING THE MAIN SEQUENCE
theta = np.zeros((2, 1))
Theta, theta_history, cost_history = gradientDecent(theta, feature, observation)
time2 = time.time()

print("VALUE OF PARAMETER USING GRADIENT DECENT: ", Theta)
print("TOTAL TIME TAKEN BY UNI-VARIATE LINEAR REGRESSION WITH GRADIENT DECENT: ", time2-time1)


'''USING NORMAL EQUATION AS OUR ALGORITHM'''
time1 = time.time()

# CONVERTING OUR DATA FRAME TO FEATURE AND OBSERVATION MATRIX
feature = np.array(food_truck_data.loc[:, ["Ones", "Population"]])
observation = np.array(food_truck_data.loc[:, ["Profit"]])


# DEFINING COST FUNCTION
def computeCostNE(theta, feature, observation):
    m = len(feature)
    cost = (1/2*m)*(np.sum(np.square(np.subtract(np.dot(feature, theta), observation))))
    return cost


# DEFINING NORMAL EQUATION FUNCTION
def normalEquation(feature, observation):
    theta = np.dot(np.dot(np.linalg.inv(np.dot(feature.T, feature)), feature.T), observation)
    cost = computeCost(theta, feature, observation)
    return theta, cost


# WRITING OUR MAIN FUNCTION
ThetaNE, cost = normalEquation(feature, observation)
time2 = time.time()

print("VALUE OF PARAMETER USING NORMAL EQUATION: ", ThetaNE)
print("TOTAL TIME TAKEN BY UNI-VARIATE LINEAR REGRESSION WITH NORMAL EQUATION: ", time2-time1)