import numpy as np
import matplotlib.pyplot as plt
from csvreaderv0 import csvreader

class NeuralNetwork:
    # initialize with random weights and bias and initialize learning rate
    def __init__(self, learning_rate):
        self.weights = []
        for i in range(6):
            self.weights.append(np.random.randn())

        self.weights = np.asarray(self.weights)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        
    # returns a prediction made based on an input vector, and network's weights and biases
    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        return layer_1
    
    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        prediction = layer_1

        # compute change in error wrt weights and bias, using chain rule for both
        derror_dprediction = 2 * (prediction - target)
        dprediction_dweights = input_vector
        dprediction_dbias = 1


        """ overflow error """
        derror_dweights = np.round(input_vector * derror_dprediction)
        derror_dbias = derror_dprediction * dprediction_dbias
        return derror_dweights, derror_dbias
    
    def _update_parameters(self, derror_dweights, derror_dbias):
        self.weights = self.weights - derror_dweights * self.learning_rate
        self.bias = self.bias - derror_dbias * self.learning_rate


    # returns an arraw with the error on every 10th piece of data 
    def train(self, input_vectors, targets, iterations, tt_split):
        
        train_split = round(len(input_vectors) * tt_split)
        
        
        # update weights based on training data
        for i in range(iterations):
            # pick a random index from the first tt_split percent of the data
            random_index = random_data_index = np.random.randint(train_split)

            input_vector = input_vectors[random_index]
            target = targets[random_index]
            
            derror_dweights, derror_dbias = self._compute_gradients(input_vector, target)
            self._update_parameters(derror_dweights, derror_dbias)

        # test on the testing data
        test_errors = []
        for i in range(train_split, len(input_vectors)):
            prediction = self.predict(input_vectors[i])
    
            # track the error every 10th iteration
            if i % 10 == 0:
                test_error = prediction - targets[i]
                test_errors.append(test_error)
        return test_errors, self.weights

# read data from csv
input_vectors, targets = csvreader("nge.csv")

# Learning rate 
# Traditional default learning rate values are 0.1, 0.01, and 0.001.
# For this model, a training rate in the order of 0.0001 works best
learning_rate = 0.0001


# Initialize a network
neural_network = NeuralNetwork(learning_rate)
testing_error, weights = neural_network.train(input_vectors, targets, 500, 0.6)


mse_test = np.square(testing_error).mean()
mean_error_test = np.sqrt(mse_test)

print(f"mean error: {mean_error_test}")
plt.plot(testing_error)
plt.xlabel("Iterations")
plt.ylabel("Error for testing instances")
plt.title(f"MSE: {mean_error_test}")
plt.savefig("testing_error.png")