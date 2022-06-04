# ann.py #

This artificial neural network features a standard neural network developed without the use of any machine learning libraries. This was an intional decision as I wanted to understand from first principles the mathematics and programming behind the processes involved (feedforward methods, backpropagation, training/testing etc.)

I have to give thanks to Michael Nielsen and his excellent walkthrough of [Using neural nets to recognize handwritten digits](http://neuralnetworksanddeeplearning.com/chap1.html). His code provided an excellent foundation and a great walkthrough on implementing vector calculus in python.


## readcsv.py ##
A read function for reading stock data from a csv is first imported from readcsv.py. The function takes as an argument the name of the csv file in the 'data/' directory that is to be read and returns an array of tuples (x, y) where x is the input vector and y is the output vector. 

The input vector consists of 6 elements which describe the stock on a given day. The descriptors were selected in accordance with those used in research conducted by [Mehar Vijha, Deeksha Chandolab, Vinay Anand Tikkiwalb and Arun Kumar (2020)](https://www.sciencedirect.com/science/article/pii/S1877050920307924) which were successfully guided an ANN in making fairly accurate predictions. These inputes are:

* high - low 
* open - close
* 7 day moving average
* 14 day moving average
* 21 day moving average
* 7 day standard deviation

The second output or target vector describe the change in closing price using three elements where the nth target describes the change in closing price between day n and day n+1. The first and second  elements indicate the likelihood of the closing price increasing or decreasing respectively while the third is the volume of the change expressed as a fraction of the original price.

The function reads the data staring from the 22nd entry in order to get a 21 day moving average and stops at the second-to-last entry because a target cannot be found for the last value.


## Train Test Split ##
Once the data is read, a train test split that suits the problem at hand can be specified and the input data can be appropriately apportioned into training and testing data.

```python
data = read("cow.csv")
tt_split = 0.6
n = len(data)
split_index = int(np.round(tt_split * n))
training_data = data[0:split_index]
test_data = data[split_index:-1]
```

## NeuralNetwork Class ##
In short, the Neural Network defined can be intialised using an array of integers such that the kth element of the array indicates the number of nodes in the kth layer of the network.

```python
#An array defining a network with 4 input nodes, 10 output nodes and one hidden layer with 6 nodes.

[4, 6, 10]
```

The operations neccessary for training and making predictions are defined as methods. The feedforward method feeds the inputs from the first layer through the hidden layers to the output layer and makes use of a sigmoid function to normalize the data as it is passes through the network. 

The NeuralNetwork class is initialized with 6 random weights - one for each element in the input vector, 1 bias for the 1 hidden layer and the learning rate

```python
import numpy as np

self.weights = []
    for i in range(6):
		self.weights.append(np.random.randn())
        self.weights = np.asarray(self.weights)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
```

The NeuralNetwork class also has a *train* method 

```python
def train(self, input_vectors, targets, iterations, tt_split):
```

* **input_vectors** - a two dimensional array. The first dimension has n elements, where n is the number of data entries and the second dimension has the afformention 6 decriptive elements.
* **targets** - the array of target values for each day
* **tt_split** - the train-test split
* **iterations** - the number of times to update weights using the training split

The train method makes a prediction on a random instance from the training split and updates the weights and biases to minimise to error as many times as there are iterations

```python
for i in range(iterations):
    random_index = random_data_index = np.random.randint(train_split)

    input_vector = input_vectors[random_index]
	target = targets[random_index]

    derivative = self._compute_gradients(input_vector, target)
    self._update_parameters(derivative)
```

It then tests the model by making predictions on the test data, keeping track of the error for every 10th prediction. It returns an array of errors.

```python
	for i in range(train_split, len(input_vectors)):
        prediction = self.predict(input_vectors[i])
        if i % 10 == 0:
            error = prediction - targets[i]
            errors.append(error)
return errors
```
That array can then be used to make a plot of the errors over the test data.

```python
import matplotlib.pyplot as plt

mse_test = np.square(testing_error).mean()
mean_error_test = np.sqrt(mse_test)

print(f"mean error: {mean_error_test}")
plt.plot(testing_error)
plt.xlabel("Iterations")
plt.ylabel("Error for testing instances")
plt.title(f"MSE: {mean_error_test}")
plt.savefig("testing_error.png")

```

## Version 0.0 ##
The weigths and the bias are updated using a highly simplified version gradient descent. A derivative is subtracted from the weights and the biases. 

Derivative = (dC/dp) * eta 

where C is the quadratic cost, p is the prediction, t is the target and eta is the learning rate

The mean error is in the range 0.4 - 1.3 


## Version 0.1 ##
The weights are updated by subtracting de/dw * eta where e is the error 

e = p - t

The bias is updated by subtracting de/db * eta where e is the error and b is the bias

This network had a poor performance. Eta had to be in the order of 0.0001. Anything lower and the network did not learn quickly enough and made very inaccurate predictions. 
Anything higher and the error oscillates between positive and negative values, getting much larger with each iteration. The program crashes very quickly because of an overflow error when this happens

The mean error is in range 0.8 - 1.3


