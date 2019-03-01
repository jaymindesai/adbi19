import statsmodels.formula.api as smf
import sklearn.metrics as sm
import pandas as pd
import numpy as np
import math
import sys

# The path to the data folder should be given as input
if len(sys.argv) != 2:
    print('bitcoin.py <path to data folder>')
    sys.exit(1)
data_path = sys.argv[1]

# Reading the vectors from the given csv files
train1_90 = pd.read_csv(data_path + '/train1_90.csv')
train1_180 = pd.read_csv(data_path + '/train1_180.csv')
train1_360 = pd.read_csv(data_path + '/train1_360.csv')

train2_90 = pd.read_csv(data_path + '/train2_90.csv')
train2_180 = pd.read_csv(data_path + '/train2_180.csv')
train2_360 = pd.read_csv(data_path + '/train2_360.csv')

test_90 = pd.read_csv(data_path + '/test_90.csv')
test_180 = pd.read_csv(data_path + '/test_180.csv')
test_360 = pd.read_csv(data_path + '/test_360.csv')


def computeDelta(wt, X, Xi):
    """
    This function computes equation 6 of the paper, but with the euclidean distance 
    replaced by the similarity function given in Equation 9.

    Parameters
    ----------
    wt : int
        This is the constant c at the top of the right column on page 4.
    X : A row of Pandas dataframe
        Corresponds to (x, y) in Equation 6.
    Xi : Panda Dataframe
        Corresponds to a dataframe of (xi, yi) in Equation 6.

    Returns
    -------
    float
        The output of equation 6, a prediction of the average price change.
    """
    n, d = 0, 0

    for _, row in Xi.iterrows():
        t = math.exp(wt * _similarity(X[:-1], row[:-1]))
        n += row[-1] * t
        d += t

    return n / d


def _similarity(a, b):
    mean_a, mean_b = a.mean(), b.mean()
    std_a, std_b = a.std(), b.std()
    a = a.apply(lambda e: e - mean_a)
    b = b.apply(lambda e: e - mean_b)
    n = a.mul(b).sum()
    d = a.size * std_a * std_b

    return n / d


def _deltaPi(before, after, constant):
    result = np.empty(0)
    for i in range(0, len(before.index)):
        result = np.append(result, computeDelta(constant, after.iloc[i], before))
    return result


# Perform the Bayesian Regression to predict the average price change for each dataset of train2 using train1 as input.
# These will be used to estimate the coefficients (w0, w1, w2, and w3) in equation 8.

weight = 2  # This constant was not specified in the paper, but we will use 2.

trainDeltaP90 = _deltaPi(train1_90, train2_90, weight)
trainDeltaP180 = _deltaPi(train1_180, train2_180, weight)
trainDeltaP360 = _deltaPi(train1_360, train2_360, weight)

# Actual deltaP values for the train2 data.
trainDeltaP = np.asarray(train2_360[['Yi']])
trainDeltaP = np.reshape(trainDeltaP, -1)

# Combine all the training data
d = {'deltaP': trainDeltaP,
     'deltaP90': trainDeltaP90,
     'deltaP180': trainDeltaP180,
     'deltaP360': trainDeltaP360}
trainData = pd.DataFrame(d)

# Feed the data: [deltaP, deltaP90, deltaP180, deltaP360] to train the linear model.
model = smf.ols(formula="deltaP ~ deltaP90 + deltaP180 + deltaP360", data=trainData).fit()

# Print the weights from the model
print(model.params)

# Perform the Bayesian Regression to predict the average price change for each dataset of test using train1 as input.
# This should be similar to above where it was computed for train2.

testDeltaP90 = _deltaPi(train1_90, test_90, weight)
testDeltaP180 = _deltaPi(train1_180, test_180, weight)
testDeltaP360 = _deltaPi(train1_360, test_360, weight)

# Actual deltaP values for test data.
testDeltaP = np.asarray(test_360[['Yi']])
testDeltaP = np.reshape(testDeltaP, -1)

# Combine all the test data
d = {'deltaP': testDeltaP,
     'deltaP90': testDeltaP90,
     'deltaP180': testDeltaP180,
     'deltaP360': testDeltaP360}
testData = pd.DataFrame(d)

# Predict price variation on the test data set.
result = model.predict(testData)
compare = {'Actual': testDeltaP,
           'Predicted': result}
compareDF = pd.DataFrame(compare)

# Compute the MSE and print the result
MSE = sm.mean_squared_error(compareDF['Actual'], compareDF['Predicted'])
print("The MSE is %f" % MSE)
