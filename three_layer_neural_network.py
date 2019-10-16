__author__ = 'matt_moon'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from scipy.special import softmax, expit

def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

########################################################################################################################
########################################################################################################################
# YOUR ASSSIGMENT STARTS HERE
# FOLLOW THE INSTRUCTION BELOW TO BUILD AND TRAIN A 3-LAYER NEURAL NETWORK
########################################################################################################################
########################################################################################################################
class NeuralNetwork(object):
    """
    This class builds and trains a neural network
    """
    def __init__(self, nn_input_dim, nn_hidden_dim , nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        
        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hidden_dim))
        self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
        self.b2 = np.zeros((1, self.nn_output_dim))

    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''

        if type.lower() == 'tanh':
            return np.tanh(z)
        elif type.lower() == 'sigmoid':
            return expit(z)
        elif type.lower() == 'relu':
            return np.maximum(np.zeros(z.shape), z)

        return None

    def diff_actFun(self, z, type):
        '''
        diff_actFun compute the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        if type.lower() == 'tanh':
            return 1 - np.square(np.tanh(z))
        elif type.lower() == 'sigmoid':
            return expit(z)*(1-expit(z))
        elif type.lower() == 'relu':
            return (1+np.sign(z))/2

        return None

    def feedforward(self, X, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        # YOU IMPLEMENT YOUR feedforward HERE

        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = actFun(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.probs = softmax(self.z2, axis=1)
        return None

    def calculate_loss(self, X, y):
        '''
        calculate_loss compute the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE

        data_loss = -sum(np.log(self.probs[np.indices(y.shape)[0], y]))

        # Add regulatization term to loss (optional)
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        return (1. / num_examples) * data_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        # IMPLEMENT YOUR BACKPROP HERE

        # dW2 = dL/dW2
        # db2 = dL/db2
        # dW1 = dL/dW1
        # db1 = dL/db1
        N = len(X)

        one_hot = np.zeros((N, self.nn_output_dim))
        one_hot[np.indices(y.shape)[0], y] = 1
        Gamma2 = self.probs - one_hot
        dW2 = (1. / N) * np.dot(self.a1.T, Gamma2)
        db2 = (1. / N) * np.sum(Gamma2, axis=0)

        Gamma1 = np.dot(Gamma2, self.W2.T) * self.diff_actFun(self.z1, self.actFun_type)
        dW1 = (1. / N) * np.dot(X.T, Gamma1)
        db1 = (1. / N) * np.sum(Gamma1, axis=0)

        return dW1, dW2, db1, db2

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            dW1, dW2, db1, db2 = self.backprop(X, y)

            # Add derivatives of regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1

            # Gradient descent parameter update
            self.W1 += -epsilon * dW1
            self.b1 += -epsilon * db1
            self.W2 += -epsilon * dW2
            self.b2 += -epsilon * db2

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plot the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)


class DeepNeuralNetwork(NeuralNetwork):
    def __init__(self, nn_input_dim, nn_hidden_dim , nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0, n_hidden=1):
        NeuralNetwork.__init__(self, nn_input_dim, nn_hidden_dim , nn_output_dim, actFun_type, reg_lambda, seed)
        self.layers = [Layer(nn_input_dim, nn_hidden_dim)] + [Layer(nn_hidden_dim, nn_hidden_dim) for i in range(n_hidden-1)] + [Layer(nn_hidden_dim, nn_output_dim)]

    def feedforward(self, X, actFun):
        layer_input = X
        for i in range(len(self.layers)):
            layer_input = self.layers[i].feedforward(layer_input, actFun)
        self.probs = softmax(layer_input, axis=1)

    def calculate_loss(self, X, y):
        N = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        data_loss = -sum(np.log(self.probs[np.indices(y.shape)[0], y]))
        for layer in self.layers:
            data_loss += self.reg_lambda / 2 * np.sum(np.square(layer.W))
        return data_loss/N

    def backprop(self, X, y):
        N = len(X)
        one_hot = np.zeros((N, self.nn_output_dim))
        one_hot[np.indices(y.shape)[0], y] = 1

        layer = self.layers[-1]
        layer.Gamma = self.probs - one_hot
        layer.dW = (1. / N) * np.dot(layer.x.T, layer.Gamma)
        layer.db = (1. / N) * np.sum(layer.Gamma, axis=0)

        for i in range(len(self.layers)-2, -1, -1):
            self.layers[i].backprop(self.layers[i+1], lambda q: self.diff_actFun(q, self.actFun_type))

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        for i in range(num_passes):
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            self.backprop(X, y)
            for layer in self.layers:
                layer.update(self.reg_lambda, epsilon)

            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))


class Layer(object):
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = np.random.randn(in_dim, out_dim)/np.sqrt(self.in_dim)
        self.b = np.zeros((1, self.out_dim))

    def feedforward(self, x, actFun):
        self.N = len(x)
        self.x = x
        self.z = np.dot(x, self.W) + self.b
        self.a = actFun(self.z)
        return self.a

    def backprop(self, next_layer, diff_actFun):
        self.Gamma = np.dot(next_layer.Gamma, next_layer.W.T) * diff_actFun(self.z)
        self.dW = (1. / self.N) * np.dot(self.x.T, self.Gamma)
        self.db = (1. / self.N) * np.sum(self.Gamma, axis=0)

    def update(self, reg_lambda, epsilon):
        self.dW += reg_lambda * self.W
        self.W += -epsilon * self.dW
        self.b += -epsilon * self.db


def main():
    # generate and visualize Make-Moons dataset
    X, y = generate_data()
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()

    #model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=100 , nn_output_dim=2, actFun_type='ReLU')
    #model.fit_model(X,y)
    #model.visualize_decision_boundary(X,y)

    deepmodel = DeepNeuralNetwork(nn_input_dim=2, nn_hidden_dim=10, nn_output_dim=2, actFun_type='relu', n_hidden=5)
    deepmodel.fit_model(X,y)
    deepmodel.visualize_decision_boundary(X,y)


if __name__ == "__main__":
    main()
