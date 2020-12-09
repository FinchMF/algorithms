
import sys
import numpy as np
from NN.utils import relu, relu_derivative, sigmoid, sigmoid_derivative

NN_arch = [

    {'input_dim': 2, 'output_dim': 25, 'activation': 'relu'},
    {'input_dim': 25, 'output_dim': 50, 'activation': 'relu'},
    {'input_dim': 50, 'output_dim': 50, 'activation': 'relu'},
    {'input_dim': 50, 'output_dim': 25, 'activation': 'relu'},
    {'input_dim': 25, 'output_dim': 1, 'activation': 'sigmoid'}

]

def init_layers(NN_arch, seed=99):

    np.random.seed(seed)

    num_layers = len(NN_arch)
    params_values = {}

    for idx, layer in enumerate(NN_arch):

        layer_idx = idx + 1

        layer_input_size = layer['input_dim']
        layer_output_size = layer['output_dim']

        params_values[f'W{layer_idx}'] = np.random.randn(layer_output_size, layer_input_size) * 0.1
        params_values[f'b{layer_idx}'] = np.random.randn(layer_output_size, 1) * 0.1

    return params_values


def single_layer_forward(A_prev, W_curr, b_curr, activation='relu'):

    Z_curr = np.dot(W_curr, A_prev) + b_curr


    if activation is 'relu':

        activation_func = relu

    elif activation is 'sigmoid':

        activation_func = sigmoid

    else:

        raise Exception('[i] Activation Not Supported')

    return activation_func(Z_curr), Z_curr


def forward(X, params_values, NN_arch):

    memory = {}

    A_curr = X

    for idx, layer in enumerate(NN_arch):

        layer_idx = idx + 1

        A_prev = A_curr

        activation_curr = layer['activation']
        W_curr = params_values[f'W{layer_idx}']
        b_curr = params_values[f'b{layer_idx}']
        A_curr, Z_curr = single_layer_forward(A_prev, W_curr, b_curr, activation_curr)

        memory[f'A{idx}'] = A_prev
        memory[f'Z{layer_idx}'] = Z_curr

    return A_curr, memory


def calculate_loss(y_hat, y):

    m = y_hat.shape[1]

    loss = -1 / m * (np.dot(y, np.log(y_hat).T) + np.dot(1 - y, np.log(1 - y_hat).T))

    return np.squeeze(loss)


def convert_prob_to_class(probs):

    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ < 0.5] = 0

    return probs_


def calculate_accuracy(y_hat, y):

    y_hat = convert_prob_to_class(y_hat)
    return (y_hat == y).all(axis=0).mean()


def single_layer_backward(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation='relu'):

    m = A_prev.shape[1]

    if activation is 'relu':

        backward_activation_func = relu_derivative

    elif activation is 'sigmoid':

        backward_activation_func = sigmoid_derivative

    else:

        raise Exception('[i] Activation Not Supported')

    dZ_curr = backward_activation_func(dA_curr, Z_curr)


    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


def backward(y_hat, y, memory, params_values, NN_arch):

    grads_values = {}

    m = y.shape[1]

    y = y.reshape(y_hat.shape)

    dA_prev = -(np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat))

    for layer_idx_prev, layer in reversed(list(enumerate(NN_arch))):

        layer_idx_curr = layer_idx_prev + 1

        activation_func_curr = layer['activation']

        dA_curr = dA_prev

        A_prev = memory[f'A{layer_idx_prev}']
        Z_curr = memory[f'Z{layer_idx_curr}']

        W_curr = params_values[f'W{layer_idx_curr}']
        b_curr = params_values[f'b{layer_idx_curr}']

        dA_prev, dW_curr, db_curr = single_layer_backward(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation_func_curr)

        grads_values[f'dW{layer_idx_curr}'] = dW_curr
        grads_values[f'db{layer_idx_curr}'] = db_curr

    return grads_values


def optimize(params_values, grads_values, NN_arch, learning_rate):

    for layer_idx, layer in enumerate(NN_arch, 1):

        params_values[f'W{layer_idx}'] -= learning_rate * grads_values[f'dW{layer_idx}']
        params_values[f'b{layer_idx}'] -= learning_rate * grads_values[f'db{layer_idx}']

    return params_values


def train(X, y, NN_arch, epochs, learning_rate, verbose=False, callbacks=None):

    params_values = init_layers(NN_arch, 2)

    losses = []
    acc = []

    x = 0
    for e in range(0, epochs):

        x += 1
        z = (f'[i] {round((x/epochs)*100, 2)}% Training Complete')
        y_hat, state = forward(X, params_values, NN_arch)

        loss = calculate_loss(y_hat, y)
        losses.append(loss)

        accuracy = calculate_accuracy(y_hat, y)
        acc.append(accuracy)

        grads_values = backward(y_hat, y, state, params_values, NN_arch)
        params_values = optimize(params_values, grads_values, NN_arch, learning_rate)

        if verbose is not True:

            sys.stdout.write('\r'+z)

        if e % 50 == 0:

            if verbose:

                print(f'Epoch {e}/{epochs} - loss: {loss} - acc: {accuracy}')
            if callbacks is not None:

                callbacks(e, params_values)


    return params_values




