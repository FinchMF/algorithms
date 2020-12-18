
import sys
import numpy as np



##########################################################
# EXAMPLE NETWORK ARCHITECTURE FOR BINARY CLASSIFICATION #
##########################################################

network = [

    {'input_dim': 2, 'output_dim': 25, 'activation': 'relu'},
    {'input_dim': 25, 'output_dim': 50, 'activation': 'relu'},
    {'input_dim': 50, 'output_dim': 50, 'activation': 'relu'},
    {'input_dim': 50, 'output_dim': 25, 'activation': 'relu'},
    {'input_dim': 25, 'output_dim': 1, 'activation': 'sigmoid'}

]


########################################
# ACTIVATION FUNCTIONS AND DERIVATIVES #
########################################


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
 


#######################################
# NETWORK STRUCTURE AND FUNCTIONALITY #
#######################################


def init_layers(network, seed=99):

    np.random.seed(seed)

    num_layers = len(network)
    params = {}

    for idx, layer in enumerate(network):

        layer_idx = idx + 1

        layer_input_size = layer['input_dim']
        layer_output_size = layer['output_dim']

        params[f'W{layer_idx}'] = np.random.randn(layer_output_size, layer_input_size) * 0.1
        params[f'b{layer_idx}'] = np.random.randn(layer_output_size, 1) * 0.1

    return params


def single_layer_forward(A_prev, W_curr, b_curr, activation='relu'):

    Z_curr = np.dot(W_curr, A_prev) + b_curr


    if activation is 'relu':

        activation_func = relu

    elif activation is 'sigmoid':

        activation_func = sigmoid

    elif activation is 'softmax':

        activation_func = softmax

    else:

        raise Exception('[i] Activation Not Supported')

    return activation_func(Z_curr), Z_curr


def forward(X, params, network):

    memory = {}

    A_curr = X

    for idx, layer in enumerate(network):

        layer_idx = idx + 1

        A_prev = A_curr

        activation_curr = layer['activation']
        W_curr = params[f'W{layer_idx}']
        b_curr = params[f'b{layer_idx}']
        A_curr, Z_curr = single_layer_forward(A_prev, W_curr, b_curr, activation_curr)

        memory[f'A{idx}'] = A_prev
        memory[f'Z{layer_idx}'] = Z_curr

    return A_curr, memory


def calculate_loss(y_hat, y):

    m = y_hat.shape[1]

    loss = -1 / m * (np.dot(y, np.log(y_hat).T) + np.dot(1 - y, np.log(1 - y_hat).T))

    return np.squeeze(loss)


def calculate_cross_entropy_loss(y_hat, y):

    loss = 0.0 
    
    for true, pred in zip(y, y_hat):
        loss += np.log(pred[true-1])

    return  -1 * loss / len(y)


def convert_prob_to_class(probs, classification='binary'):

    if classification == 'binary':

        probs_ = np.copy(probs)
        probs_[probs_ > 0.5] = 1
        probs_[probs_ < 0.5] = 0

    elif classification == 'multi':

        probs_ = np.copy(probs)
        probs_ = np.argmax(probs_)

    else:

        raise Exception(f'[!] Classificaiton type {classification} Not Supported')

    return probs_


def calculate_accuracy(y_hat, y, classification='binary'):

    y_hat = convert_prob_to_class(y_hat, classification)
    return (y_hat == y).all(axis=0).mean()


def single_layer_backward(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation='relu'):

    m = A_prev.shape[1]

    if activation is 'relu':

        backward_activation_func = relu_derivative

    elif activation is 'sigmoid':

        backward_activation_func = sigmoid_derivative

    elif activation is 'softmax':

        backward_activation_func = softmax_derivative

    else:

        raise Exception('[i] Activation Not Supported')

    dZ_curr = backward_activation_func(dA_curr, Z_curr)


    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


def backward(y_hat, y, memory, params, network):

    grads = {}

    m = y.shape[1]

    y = y.reshape(y_hat.shape)

    dA_prev = -(np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat))

    for layer_idx_prev, layer in reversed(list(enumerate(network))):

        layer_idx_curr = layer_idx_prev + 1

        activation_func_curr = layer['activation']

        dA_curr = dA_prev

        A_prev = memory[f'A{layer_idx_prev}']
        Z_curr = memory[f'Z{layer_idx_curr}']

        W_curr = params[f'W{layer_idx_curr}']
        b_curr = params[f'b{layer_idx_curr}']

        dA_prev, dW_curr, db_curr = single_layer_backward(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation_func_curr)

        grads[f'dW{layer_idx_curr}'] = dW_curr
        grads[f'db{layer_idx_curr}'] = db_curr

    return grads


def optimize(params, grads, network, learning_rate):

    for layer_idx, layer in enumerate(network, 1):

        params[f'W{layer_idx}'] -= learning_rate * grads[f'dW{layer_idx}']
        params[f'b{layer_idx}'] -= learning_rate * grads[f'db{layer_idx}']

    return params


def train(X, y, network, epochs, learning_rate, classification='binary', verbose=False, callbacks=None):

    params = init_layers(network, 2)

    losses = []
    acc = []

    x = 0
    for e in range(0, epochs):

        x += 1
        z = (f'[i] {round((x/epochs)*100, 2)}% Training Complete')
        
        y_hat, state = forward(X, params, network)

        if classification == 'binary':
            loss = calculate_loss(y_hat, y)
        
        elif classification == 'multi':
            loss = calculate_cross_entropy_loss(y_hat, y)

        else:
            raise Exception(f'[i] Classificaiton Type: {classification} Not Supported. Please choose between - binary - or - multi -')

        losses.append(loss)
        

        accuracy = calculate_accuracy(y_hat, y, classification)
        acc.append(accuracy)

        grads = backward(y_hat, y, state, params, network)
        params = optimize(params, grads, network, learning_rate)

        if verbose is not True:

            sys.stdout.write('\r'+z)

        if e % 50 == 0:

            if verbose:

                print(f'Epoch {e}/{epochs} - loss: {loss} - acc: {accuracy}')
            if callbacks is not None:

                callbacks(e, params)


    return params




    