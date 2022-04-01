import numpy as np


class LogReg:

    def __init__(self, batch_size=25, num_steps=350, lr=1e-1):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.lr = lr

    def fit(self, X, Y):
        features_count = X.shape[1]

        w = np.random.randn(features_count)[:, None]
        n_objects = len(X)

        for i in range(self.num_steps):
            sample_indices = np.random.randint(0, n_objects, size=self.batch_size)

            prediction = np.dot(X[sample_indices], w)
            sigmoid_comp = 1 / (1 + np.exp((-1) * prediction))
            true_minus_pred_comp = sigmoid_comp - Y[sample_indices]

            w -= self.lr * ((np.dot(X[sample_indices].T, true_minus_pred_comp)) / self.batch_size)

        self.w = w
        return self

    def predict(self, X):
        return np.sign(X @ self.w)  # sign(matmul)
