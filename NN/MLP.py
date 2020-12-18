
import numpy as np
import NN.utils as utils



class MLP:

    def __init__(self, network):

        self.network = network

    def train(self, X_train, y_train, epochs, learning_rate, classification='binary', verbose=False, callbacks=None):

        weights = utils.train(X=np.transpose(X_train),
                                y=np.transpose(y_train.reshape((y_train.shape[0], 1))),
                                network=self.network,
                                epochs=epochs,
                                learning_rate=learning_rate,
                                classification=classification,
                                verbose=verbose,
                                callbacks=callbacks)

        return weights


    def predict(self, X_test, weights):

        preds, _ = utils.forward(X=np.transpose(X_test),
                                 params=weights,
                                 network=self.network)

        return preds



    def network_accuracy(self, preds, y_test, classification='binary'):

        acc = utils.calculate_accuracy(y_hat=preds,
                                       y=np.transpose(y_test.reshape((y_test.shape[0], 1))),
                                       classification=classification)

        print(f'Network on Accuracy: {acc}')

        return acc