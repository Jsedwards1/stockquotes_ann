# ann.py #

This artificial neural network features a standard neural network developed without the use of any machine learning libraries. This was an intional decision as I wanted to understand from first principles the mathematics and programming behind the processes involved (feedforward, backpropagation, training/testing etc.)

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
The neural network is dynamically generated and initializes random weights and biases.It is intialised using an array of integers such that the kth element of the array indicates the number of nodes in the kth layer of the network.

```python
#An array defining a network with 4 input nodes, 10 output nodes and one hidden layer with 6 nodes.

[4, 6, 10]
```

The operations neccessary for training and making predictions are defined as methods. The feedforward method feeds the inputs from the first layer through the hidden layers to the output layer and makes use of a sigmoid function to normalize the data as it is passes through the network. The backprop, SGD, cost_derivative, and update_mini_batch methods are all used in tandem to adjust the weights and biases of the network using backpropagation and stochastic gradient descent accodring to a quadratic cost function. If test data is provided to the SGD method, the accuracy of the network can be determined using the evaluate method.


## stock_net.py ##
stock_net.py implements a model with one hidden layer that contains 4 nodes - the design of the hiddel layers was aritrarily selected. It then adjusts the network's weights and biases using the training data and a learning rate of 0.1. Finally, the predict_price function is designed to be called once the network has been trained. It takes two inputs: an input vector with the 6 descriptors of a stock on a given day, and the closing price for that day and converts the 3 element output array into an actual predicted price. This is simply a utility function that helps to make sense of the predictions made by a trained network.
