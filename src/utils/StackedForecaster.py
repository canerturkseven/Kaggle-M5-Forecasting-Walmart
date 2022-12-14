import numpy as np


class StackedForecaster:
    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        for estimator, x in zip(self.estimators, X):
            estimator.fit(x, y)
        return self

    def predict(self, X):
        prediction = []
        for estimator, x in zip(self.estimators, X):
            prediction.append(estimator.predict(x))
        return np.vstack(prediction)
