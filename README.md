# Version 0 #

ANN v0 features a simple neural network model and a simple algorithm for updating weights and biases. 
There is 1 hidden layer and 1 node in the output layer which is the change in closing price between day n and day n+1


## Reading Data ##
The csv function csvreader is first imported. Ittakes as an argument the name of the csv file in the data/ directory that is to be read. The csvreader function returns 2 numpy arrays.
	
The first is a two dimensional array. The first dimension has n elements, where n is the number of data entries and the second dimension has 6 elements. These 6 elements describe the stock on a given day and were selected *(per the paper I read)*

* high - low 
* open - close
* 7 day moving average
* 14 day moving average
* 21 day moving average
* 7 day standard deviation

The second array is an array of targets. The nth target is the change in closing price between day n and day n+1

The function reads the data staring from the 22nd entry in order to get a 21 day moving average and stops at the second-to-last entry because a target cannot be found for the last value

```python
# read data from csv
input_vectors, targets = csvreader("nge.csv")
```


## NeuralNetwork Class ##
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


