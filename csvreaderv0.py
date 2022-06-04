""" 
Reading training data from a csv file

Reading starts on the 22nd entry to get a 21 day moving average

The 'target' for a datapoint depends on the entry after that datapoint so this reads up to the 
second-to-last entry rather than the last one

"""
from csv import reader
import numpy as np



def csvreader(file):
    input_vectors = []

    """
    an array of arrays storing stock data
        high - low
        close - open
        close
        7 day MA
        14 day MA
        21 day MA
        7 day std deviation
    """
    
    # Stores close prices
    closePrices = []
    targets = []
    
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
                hminl = float(row[2]) - float(row[3])
                # close - open price
                cmino = float(row[4]) - float(row[1])

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

                # 7 day std deviation
                stdDev = np.std(closePrices[dataIndex - 7:-1])
            
                # update input_vectors
                input_vectors.append([hminl, cmino, DMA_7, DMA_14, DMA_21, stdDev])

                
                # set targets to be close price for day n+1 - close price for day n
                # do it by appending 0 on the ith pass
                # then assign the difference to the ith spot on the i+1th pass
                
                if skipfirst:
                    targets.append(0)
                    skipfirst = False
                else:  
                    targets.append(0)
                    targets[dataIndex - 22] = closePrices[dataIndex] - closePrices[dataIndex - 1]
                    
                
            dataIndex+=1
            
          
    # TODO chop off the last entry of input_vectors and targets
    extra_input = input_vectors.pop(dataIndex - 22)
    extra_target = targets.pop(dataIndex - 22)

    return np.asarray(input_vectors), np.asarray(targets)


