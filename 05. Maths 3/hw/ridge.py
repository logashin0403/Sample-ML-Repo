from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


class LinRegRidge(BaseEstimator, RegressorMixin):

    def __init__(self, batch_size=25, num_steps=350, lr=1e-2):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.lr = lr

    def fit(self, X, Y, lmbd):
        features_count = X.shape[1]

        w = np.random.randn(features_count)[:, None]
        n_objects = len(X)

        for i in range(self.num_steps):
            sample_indices = np.random.randint(0, n_objects, size=self.batch_size)

            prediction = np.dot(X[sample_indices], w)
            scalar_prod = np.dot(X[sample_indices].T, prediction - Y[sample_indices])
            regularization_comp = (2 * lmbd * w) / features_count

            w -= (2 * self.lr * scalar_prod) / self.batch_size + regularization_comp

        self.w = w
        return self

    def predict(self, X):
        return X @ self.w  # matmul
