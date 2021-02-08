import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


def set_connection(username, password, host, port, db_name):  # SET UP A CONNECTION WITH THE DATA BASE AND RETURN THE CONNECTION OBJECT
    return create_engine('mysql+pymysql://'+username+":"+password+"@"+host+":"+port+"/"+db_name).connect()


def load_data(table_name,
              db_name='testdb',
              username='root',
              password='12345',
              host='localhost',
              port='3306'):  # THIS FUNCTION LOADS DATA FROM A PARTICULAR TABLE IN A DATA BASE
    query = 'SELECT * FROM '+table_name
    return pd.read_sql(query, con=set_connection(username, password, host, port, db_name))


admission_data = load_data('admission_data')
admission_data = np.array(admission_data)


def feature_map(x1, x2):  # TAKE TWO ARRAYS AND MAP THEM INTO N DIMENSIONAL MATRIX DEPENDS ON THE VALUE OF DEGREE
    degree = 2
    out = np.ones((x1.shape[0], 1))
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(x1, i-j),
                                              np.power(x2, j))))
    return out


def sigmoid(vector):  # TAKES A NUMBER AS AN INPUT AND RETURNS VALUE BETWEEN 0 AND 1
    return 1/(1+np.exp(-vector))


def cost(theta, x, y, lamb_da):  # DETERMINES THE COST(THIS FUNCTION ACTS AS AN INPUT FOR OPTIMISER FUNCTION)
    m = x.shape[0]
    cost = (-1/m)*(np.add(np.dot(y.T, np.log(sigmoid(np.dot(x, theta)))),
                          np.dot((1-y).T, np.log(1-sigmoid(np.dot(x, theta))))))
    cost += (lamb_da/(2*m))*np.dot(theta.T, theta)
    return cost


def gradient(theta, x, y, lamb_da):  # DETERMINES THE SLOPE OR GRADIENT(THIS FUNCTION ACTS AS AN INPUT FOR OPTIMISER FUNCTION)
    m = x.shape[0]
    grad = (1/m)*(np.dot(x.T, np.subtract(sigmoid(np.dot(x, theta)), y)))
    grad[1:] = grad[1:] + (lamb_da/m)*theta[1:]
    return grad


def predict(x):  # TAKES A TRAINING DATA SET AND PERFORM LOGISTIC REGRESSION ON IT(THIS FUNCTION IS THE NERVE CENTRE FOR THE WHOLE PROGRAM. OUTPUT:OUTPUT VALUE EITHER 0 OR 1/PARAMETERS/ACCURACY OF THE MODEL)
    lamb_da = 1
    theta_initial = np.zeros((x.shape[1], 1))
    parameter = opt.fmin_bfgs(f=cost,
                              x0=theta_initial,
                              fprime=gradient,
                              args=(feature_map(admission_data[:, [0]], admission_data[:, [1]]),
                                    admission_data[:, [2]].flatten(),
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
        if output[i] == admission_data[i, [2]]:
            counter += 1
    return output, parameter, (counter/len(output))*100


output, parameter, accuracy = predict(feature_map(admission_data[:, [0]], admission_data[:, [1]]))
print(accuracy)


# VISUALISING THE DATA
def feature_map_for_plot(x1, x2):
    degree = 2
    out = np.ones(1)
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(x1, i-j), np.power(x2, j))))
    return out


p = np.linspace(25, 100, 50)
q = np.linspace(25, 100, 50)
r = np.zeros((len(p), len(q)))

for i in range(len(p)):
    for j in range(len(q)):
        r[i, j] = np.dot(feature_map_for_plot(p[i], q[j]), parameter)

admitted = admission_data[admission_data[:, 2] == 1]
not_admitted = admission_data[admission_data[:,  2] == 0]
plt.scatter(admitted[:, [0]], admitted[:, [1]], s=40, label="Admitted", marker="+", color="Blue")
plt.scatter(not_admitted[:, [0]], not_admitted[:, [1]], s=40, label="Not Admitted", marker="*", color="Red")
plt.contour(p, q, r, 0)
plt.title("Admitted Vs Rejected")
plt.xlabel("Test 1 Scores")
plt.ylabel("Test 2 Scores")
plt.grid()
plt.show()