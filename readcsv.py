""" 
Reading training data from a csv file

Reading starts on the 22nd entry to get a 21 day moving average

The 'target' for a datapoint depends on the entry after that datapoint so this reads up to the 
second-to-last entry rather than the last one

The function returns the array 'data', which is a list of tuples (x, y) where x is the input vector and y is the target vector. Both vectors are stored as numpy arrays to facilitate matrix manipulation

"""
from csv import reader
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def read(file):
    data = []
    
    # Stores close prices
    closePrices = []
    

    # Temporarily stores current targets before they are pushed to the targets array
    buffer = []

    # Input parameters hi - low and close - open
    hminl = 0
    cmino = 0

    # Variables for storing moving averages and STD deviation
    dataIndex = 0
    DMA_7 = 0
    DMA_14 = 0
    DMA_21 = 0
    stdDev = 0

    with open("data/" + file, "r") as csvfile:
        trainData = reader(csvfile, delimiter=",")
        next(trainData)
        skipfirst = True
        for row in trainData:
            closePrices.append(float(row[4]))


            # start storing data once there can be a 21 day average
            if dataIndex >= 21:
                
                # high - low price
                high = float(row[2])
                low = float(row[3])
                hminl = 100 * ((high - low) / low)
                hminl = sigmoid(hminl)

                # close - open price
                cp = float(row[4])
                op = float(row[1])
                cmino = 100 * ((cp - op) / cp)
                cmino = sigmoid(cmino)

                # Moving Averages
                DMA_7 = 0
                DMA_14 = 0
                DMA_21 = 0
                for i in range(dataIndex - 21, dataIndex):
                    if i >= dataIndex - 7:
                        DMA_7 += closePrices[i]
                    if i >= dataIndex - 14:
                        DMA_14 += closePrices[i]
                    
                    DMA_21 += closePrices[i]
                
                DMA_7 = DMA_7 / 7
                DMA_14 = DMA_14 / 14
                DMA_21 = DMA_21 / 21

                DMA_7 = sigmoid(100 * np.absolute(closePrices[dataIndex] / DMA_7 - 1))
                DMA_14 = sigmoid(100 * np.absolute(closePrices[dataIndex] / DMA_14 - 1))
                DMA_21 = sigmoid(100 * np.absolute(closePrices[dataIndex] / DMA_21 - 1))

                # 7 day std deviation
                stdDev = np.std(closePrices[dataIndex - 7:-1])

                x = np.array([[hminl], [cmino], [DMA_7], [DMA_14], [DMA_21], [stdDev]])
                
                if skipfirst:
                    skipfirst = False
                    x_buffer = x
                else:  
                    absdifference = np.absolute(closePrices[dataIndex] - closePrices[dataIndex - 1])
                    percentdiff = (absdifference / closePrices[dataIndex - 1]) * 100

                    # If the closing price goes up, push a 1. If it goes down, push a 0
                    if closePrices[dataIndex] > closePrices[dataIndex - 1]:
                        #direction = 1
                        up = 1
                        down = 0
                    else:
                        #direction = 0
                        up = 0
                        down = 1

                    y = np.array([[up], [down], [sigmoid(percentdiff)]])
                    t = (x_buffer, y)
        
                    
                    x_buffer = x
                    data.append(t)
            dataIndex+=1
    return data