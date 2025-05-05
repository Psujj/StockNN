import PythonScripts.RandomNumber as Rd
import yfinance as finance
import numpy as np

# create data set: 
def create_dataset():
    data = finance.download("AAPL", '2015-01-03', '2025-01-02')
    features = data[['Close', 'Volume']].values
    ntraining = 2299
    trainingFeatures = features[:ntraining]
    testFeatures = features[ntraining:]

    # normalzie the data  
    trainMax = np.max(trainingFeatures, axis=0) # Get max of each column
    trainMax[trainMax == 0] = 1e-8 # Avoid divide-by-zero
    trainingFeatures = trainingFeatures / trainMax # Normalize
    testNormalized = testFeatures / trainMax

    # indexes:
    minIndex = 14
    return trainingFeatures, testNormalized


# Generate random indexes with a seed:
def createInput(Training_Data, index):
    if index < 14:
        return 
    InputArray = Training_Data[index - 14:index]
    ExpectedCloseOutput = Training_Data[index][0] # only want the predicted close price
    return InputArray, ExpectedCloseOutput




