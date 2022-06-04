import numpy as np
import random

class Network:
    # Initialise the network with random weights and biases
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(y, x) for x, y in zip (sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

    def feedforward(self, a):
        """Return the output of the network if a is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic gradient descent. The training_data is a list of tuples (x, y) representing the training inputs and the desired outputs. The other non-optional parameters are self-explanatory.  
        
        If test_data is provided then the
        network will be evaluated against the test data after each epoch, and partial progress printed out. This is useful for tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                dir_accuracy, cp_error = self.evaluate(test_data)
                print("Epoch {0}: ".format(j))
                print("Direction of close price prediction: {0} / {1}".format(dir_accuracy, n_test))
                print("Mean error in stock price prediction: {0}".format(cp_error))
            else:
                print("Epoch {0} complete".format(j))
    
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch. The 'mini_batch' is a list of tuples (x, y), and 'eta' is the learning rate."""

        """nabla_b and nabla_w store the chages in bias and weights respectively that a single training example wants to make. The changes each training example wants to make to each weight is summed up and stored here"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        """Return a tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x. nabla_b and
        nabla_w are layer-by-layer lists of numpy arrays, similar to self.biases and self.weights."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # backward pass

        # delta is our error for the output layer
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x \partial a for the output activations."""
        return (output_activations-y)

    def evaluate(self, test_data):
        """Evaluates the network using the test data provided. Returns the number of correct predictions for the direction in which a stock moved - y[0] and the mean error in magnitude of movement - y[1]
        """
        dir_accuracy = 0
        error = 0
        for (x, y) in test_data:
            prediction = self.feedforward(x)
            if prediction[0] > prediction[1]:
                up = 1
                down = 0
            else:
                up = 0
                down = 1

            if up == y[0]:
                dir_accuracy += 1
            """
            if int(np.round(prediction[0])) == y[0]:
                dir_accuracy += 1
            """
            error =  error + np.absolute((prediction[2] - y[2]))
            cp_error = error / len(test_data)

        return dir_accuracy, cp_error

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))