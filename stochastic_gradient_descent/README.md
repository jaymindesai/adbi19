###### Jaymin Desai, _jddesai2@ncsu.edu_ 

## Requirements

 - Python 3
 - Numpy
 - Pandas
 
## Running SGD

`python sgd.py`

Running this file will produce the following output:


Ground Truth: [160. 170. 180. 190. 200.]

Predictions: [160.03 170.06 180.09 190.12 200.15]

RMSE: 0.10728658957938565


The data used is dummy data and the code works fine with any dataset; replace _train.csv_ and _test.csv_ inside `/data` folder with respective files.
It was difficult to write test cases because of the stochasticity, so I have printed RMSE to console instead.

##### Hyperparameters

Change the values of hyperparameters in the driver code inside _sgd.py_