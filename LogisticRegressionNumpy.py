import numpy as np
import matplotlib.pyplot as plt

def load_data():
    points = np.genfromtxt('Microchip_testing.txt', delimiter = ',')
    X = points[:,0:2]
    y = points[:,2]
    y = y.reshape((len(y),1))
    return X.T, y.T

def map_feature(X1, X2, shape, degree = 6):
    new_X = np.ones((degree, shape))
    new_X[0,:] = (X1)
    new_X[1,:] = (X2)
    new_X[2,:] = (X1**2)
    new_X[3,:] = (X2**2)
    new_X[4,:] = (X1*X2)
    new_X[5,:] = ((X1*X2)**2)
    return new_X

def plot_data(X, y):
    accepted = plt.scatter(X[0, np.where(y == 1)], X[1, np.where(y == 1)], color='blue', label="Accepted",)
    not_accepted = plt.scatter(X[0, np.where(y == 0)], X[1, np.where(y == 0)], color='red', label="Not Accepted")
    plt.title('Microchip Evaluation')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(handles=[accepted, not_accepted], loc = 3)
    plt.show()

def initialize_with_zeros(dim):
    W = np.zeros(shape=(dim, 1))
    b = 0
    return W, b

def predict(W, b, X):

    m = X.shape[1]
    y_hat = np.zeros((1,m))
    W = W.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(W.T, X) + b)
    
    for i in range(A.shape[1]):
        
        if A[0,i] > 0.5:
            y_hat[0,i] = 1
        else:
            y_hat[0,i] = 0
    return y_hat

def sigmoid(Z):
    A =  1 / (1 + np.exp(-Z))    
    return A

def propagate(W, b, X, y):
    m = X.shape[1]
    A = sigmoid(np.dot(W.T, X) + b)

    cost = - (1 / m) * np.sum((y * np.log(A)) + ((1 - y) * np.log(1 - A)))

    dw = (1 / m) * np.dot(X, (A - y).T)
    db = (1 / m) * np.sum(A - y)

    cost = np.squeeze(cost)

    grads = {"dw": dw,
             "db": db}
    return grads, cost

def optimize(W, b, X, y, num_iterations, learning_rate):
    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(W, b, X, y)

        dw = grads["dw"]
        db = grads["db"]

        W = W - learning_rate * dw
        b = b - learning_rate * db
        
        if i % 100 == 0:
            plot_predicted_data(X, y, W, b)
            costs.append(cost)
    
    params = {"W": W,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs

def model(X, y, num_iterations = 10000, learning_rate = 0.1):    
    W, b = initialize_with_zeros(X.shape[0])

    parameters, grads, costs = optimize(W, b, X, y, num_iterations, learning_rate)
    W = parameters["W"]
    b = parameters["b"]

    y_hat = predict(W, b, X)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_hat - y)) * 100))
    
    d = {"costs": costs,
         "y_hat" : y_hat,
         "W" : W, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d

def plot_cost_history(cost_history):
    plt.figure(2)
    plt.plot(cost_history, color = 'blue')
    plt.title('Cost Reduction by Gradient Descent')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

def plot_predicted_data(X, y, W, b):
    plt.clf()
    accepted = plt.scatter(X[0, np.where(y == 1)], X[1, np.where(y == 1)], color='blue', label="Accepted",)
    not_accepted = plt.scatter(X[0, np.where(y == 0)], X[1, np.where(y == 0)], color='red', label="Not Accepted")
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    
    z = np.zeros((len(u), len(v)))
    for i in range (len(u)):
        for j in range (len(v)):
            z[i,j] = np.dot(W.T, map_feature(u[i], v[j], 1)) + b
    z = z.T
    plt.contour(u, v, z, 0)
    plt.title('Microchip Evaluation')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(handles=[accepted, not_accepted], loc = 3)
    plt.draw()
    plt.pause(0.00001)

def run():
    X, y = load_data()
    X = map_feature(X[0,:], X[1,:], X.shape[1])
    d = model(X, y)
    plot_predicted_data(X, y, d["W"], d["b"])
    plot_cost_history(d["costs"])

if __name__ == '__main__':
    run()