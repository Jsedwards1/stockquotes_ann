import numpy as np
from readcsv import read
import ann

# read data and split it into training and testing datasets according to the train-test split, tt_split
data = read("cow.csv")
tt_split = 0.6
n = len(data)
split_index = int(np.round(tt_split * n))
training_data = data[0:split_index]
test_data = data[split_index:-1]

net = ann.Network([6, 4, 3])
net.SGD(training_data, 5, 15, 0.1)


""" Outputs the closing price for day n+1 given x - the input data for day n as a numpy array and cp - the closing price for day n

prediction is of shape 2x1 i.e. prediction = [[direction], [amount]]. So prediction[1][0] returns a value whereas prediction[1] than a list containing one value"""
def predict_price(x, cp):
    prediction = net.feedforward(x)
    if int(np.round(prediction[0])) == 0:
        
        return cp - prediction[1][0]
    else:
        return cp + prediction[1][0]
