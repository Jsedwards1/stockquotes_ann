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
net.SGD(training_data, 5, 15, 5)


""" Outputs the closing price for day n+1 given x - the input data for day n as a numpy array and cp - the closing price for day n

prediction is of shape 2x1 i.e. prediction = [[direction], [amount]]. So prediction[1][0] returns a value whereas prediction[1] than a list containing one value"""
def predict_price(x, cp):
    prediction = net.feedforward(x)
    if int(np.round(prediction[0])) == 0:
        
        return cp - prediction[1][0]
    else:
        return cp + prediction[1][0]

def get_prediction(x):
    return net.feedforward(x)
    
pred_sum = 0
preds = []
for i in range(0, 10):
    pred_sum = pred_sum + get_prediction(data[i][0])[2][0]
    preds.append(get_prediction(data[i][0])[2][0])
    avg = pred_sum / 10
    stdDev = np.std(preds)

print(f"Average: {avg}")
print(f"StdDev: {stdDev}")
