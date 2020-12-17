
import numpy as np


def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def softmax(z):
		e = np.exp(float(z))
		return (e/np.sum(e))

def sigmoid_derivative(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_derivative(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def softmax_derivative(dA, Z):
    soft = softmax(Z)
    return dA * soft * (1 - soft)
 






    