import numpy as np
import pandas as pd


class SGD:
    """
    Stochastic Gradient Descent for Linear Regression with L2 Regularization
    """

    def __init__(self, learning_rate=0.001, l2_lambda=0.1, batch_size=1, epochs=50):
        self.W = None  # beta vector
        self.epochs = epochs  # no. of epochs
        self.learning_rate = learning_rate  # learning rate: alpha
        self.batch_size = batch_size  # batch size to compute gradients over at a time
        self.l2_lambda = l2_lambda  # L2 regularization coeff: lambda

    @staticmethod
    def stack_ones(X):
        """Stack a column vector of 1's to X; beta_0 coeffs for clean dot product"""
        return np.hstack((np.ones((X.shape[0], 1)), X))

    @staticmethod
    def rmse(y_true, y_pred):
        """Compute root mean squared error"""
        return np.sqrt(np.square(y_true - y_pred).mean())

    def fit(self, X, y):
        """Compute gradients and train weights (beta_1, beta2, ..., beta_n) and biases (beta_0)"""
        X = self.stack_ones(X)
        self.W = np.random.normal(0, 1, (X.shape[1], 1))  # initialize weights from normal distribution; mu = 0, sd = 1
        for i in range(self.epochs):
            for j in range(X.shape[0] // self.batch_size):
                idx = np.random.choice(X.shape[0], self.batch_size, replace=False)  # introduce stochasticity
                self.W -= self.learning_rate * self._compute_gradient(X[idx, :], y[idx])  # update gradients batchwise
        return self

    def predict(self, X):
        X = self.stack_ones(X).T
        return np.squeeze(self.W.T.dot(X))

    def _compute_gradient(self, X, y):
        gradient = np.zeros((X.shape[1], 1))  # initialize gradients as 0's
        for xi, yi in zip(X, y):
            gradient += np.reshape((self.W.T.dot(xi) - yi) * xi, (X.shape[1], 1))  # accumulate gradients across samples
        gradient *= 2 / X.shape[0]  # normalize over batch size
        regularization = 2 * self.l2_lambda * self.W  # l2 regularization
        return gradient + regularization


if __name__ == '__main__':
    np.random.seed(100)

    train_data = pd.read_csv('data/train.csv').astype(float)  # dummy training data
    test_data = pd.read_csv('data/test.csv').astype(float)  # dummy testing data

    X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

    # Hyperparameters:
    learning_rate = 0.001
    l2_lambda = 0.1
    batch_size = 1
    epochs = 50

    y_pred = SGD(learning_rate, l2_lambda, batch_size, epochs).fit(X_train, y_train).predict(X_test)

    print('Ground Truth: {}'.format(y_test.values))
    print('Predictions: {}'.format((y_pred * 100).astype(int) / 100))
    print('RMSE: {}'.format(SGD.rmse(y_test, y_pred)))
