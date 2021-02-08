import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


def set_connection(username, password, host, port, db_name):
    return create_engine('mysql+pymysql://'+username+":"+password+"@"+host+":"+port+"/"+db_name).connect()


def load_data(table_name,
              db_name='testdb',
              username='root',
              password='12345',
              host='localhost',
              port='3306'):
    query = 'SELECT * FROM '+table_name
    return pd.read_sql(query, con=set_connection(username, password, host, port, db_name))


microchip_data = load_data('microchip_data')
microchip_data = np.array(microchip_data)  # CONVERTING THE DATA FRAME INTO AN N DIMENSIONAL MATRIX


def feature_map(x1, x2):
    degree = 6
    out = np.ones((x1.shape[0], 1))
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(x1, i-j), np.power(x2, j))))
    return out


def sigmoid(vector):
    return 1/(1+np.exp(-vector))


def cost(theta, x, y, lamb_da):
    m = x.shape[0]
    cost = ((-1/m)*(np.add(np.dot(y.T, np.log(sigmoid(np.dot(x, theta)))),
                           np.dot((1-y).T, np.log(1-sigmoid(np.dot(x, theta)))))))
    cost += (lamb_da/(2*m))*(np.dot(theta.T, theta))
    return cost


def gradient(theta, x, y, lamb_da):
    m = x.shape[0]
    grad = (1/m)*(np.dot(x.T, np.subtract(sigmoid(np.dot(x, theta)), y)))
    grad[1:] = grad[1:] + (lamb_da/m)*theta[1:]
    return grad


def predict(x):
    lamb_da = 1  # IF 0(MODEL WILL OVER FIT THE DATA(GOOD FOR FITTING TRAINING DATA BUT FAILS TO PREDICT))/IF >0 (MODEL WILL DECENTLY FIT THE DATA(DOESN'T FIT DATA WELL BUT GOOD FOR PREDICTION)/ IF >5 (MODEL WILL UNDERFIT THE DATA))
    theta_initial = np.zeros((x.shape[1], 1))
    parameter = opt.fmin_bfgs(f=cost,
                              x0=theta_initial,
                              fprime=gradient,
                              args=(feature_map(microchip_data[:, [0]], microchip_data[:, [1]]),
                                    microchip_data[:, [2]].flatten(),
                                    lamb_da))
    parameter = np.array(parameter).reshape(x.shape[1], 1)
    output = sigmoid(np.dot(x, parameter))
    for i in range(0, len(output)):
        if output[i] >= 0.5:
            output[i] = 1
        else:
            output[i] = 0
    counter = 0
    for i in range(0, len(output)):
        if output[i] == microchip_data[i, [2]]:
            counter += 1
    return output, parameter, (counter/len(output))*100


output, parameter, accuracy = predict(feature_map(microchip_data[:, [0]], microchip_data[:, [1]]))
print(accuracy)


# VISUALISING THE DATA
def feature_map_for_data(x1, x2):
    degree = 6
    out = np.ones(1)
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(x1, i-j),
                                              np.power(x2, j))))
    return out


p = np.linspace(-1, 1.5, 50)
q = np.linspace(-1, 1.5, 50)
r = np.zeros((len(p), len(q)))

for i in range(len(p)):
    for j in range(len(q)):
        r[i, j] = np.dot(feature_map_for_data(p[i], q[j]), parameter)

accepted = microchip_data[microchip_data[:, 2] == 1]
rejected = microchip_data[microchip_data[:, 2] == 0]
plt.scatter(accepted[:, [0]], accepted[:, [1]], s=40, label="Accepted", marker="+", color="Blue")
plt.scatter(rejected[:, [0]], rejected[:, [1]], s=40, label="Rejected", marker="*", color="Red")
plt.contour(p, q, r, 0)
plt.title("Accepted vs Rejected (Over-Fitted Model)")
plt.xlabel("Microchip Test 1")
plt.ylabel("Microchip Test 2")
plt.grid()
plt.show()