import math
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from itertools import product
from scipy.spatial import distance


def epanechnikov(dist):
    return 3.0 / 4 * (1 - dist ** 2) if abs(dist) <= 1.0 else 0.0


def quadratic(dist):
    return 15.0 / 16 * (1 - dist ** 2) if abs(dist) <= 1.0 else 0.0


def triangular(dist):
    return 1 - abs(dist) if abs(dist) <= 1.0 else 0.0


def gauss(dist):
    return math.pow(2 * np.pi, -1.0 / 2) * np.exp(-1 / 2 * (dist ** 2))


def rect(dist):
    return 1 / 2 if abs(dist) <= 1.0 else 0.0


class MPF:
    """ Реализация метода потенциальных функций """
    kernels = {
        "epanechnikov": epanechnikov,
        "quadratic": quadratic,
        "triangular": triangular,
        "gauss": gauss}
    
    
    def __init__(self,H = 5,kernel = "gauss", metric_power = 2):
        self.H = H
        self.kernel = self.kernels[kernel]
        self.metric_power = metric_power
                                
    
    def predict_array(self, X):
        return np.array([self.predict(i) for i in X])
    
    def predict_array_broadcast(self, X):
        return np.array([self.predict_broadcast(i) for i in X])
    
    def predict(self, X):
        pred_array = np.zeros_like(np.unique(self.train_y), dtype=np.float)
        for i in range(len(self.train_X)):
            label = self.train_y[i]
            dist = distance.minkowski(X,self.train_X[i],self.metric_power)
            pred_array[label] += self.potentials[i] * self.kernel(dist / self.H)
        return np.argmax(pred_array)
    
    def predict_broadcast(self, X):
        test_x = np.copy(X)
        test_x = test_x[np.newaxis, :]
        diff = test_x[:, np.newaxis, :] - self.train_X[np.newaxis, :, :]
        distances = np.sqrt(np.sum((diff ** 2), -1))
        weights = self.potentials * self.kernel(distances / self.H)
        classes = np.unique(self.train_y) 
        pred_array = np.zeros((test_x.shape[0], len(classes)))            
        for c in classes:
            pred_array[:, c] = np.sum(weights[:, self.train_y == c], axis=1)
        return np.argmax(pred_array, axis=1)
        
    def fit(self, X, y, iterations = 10):
        assert X.shape[0] == y.shape[0]
        self.train_X = np.copy(X)
        self.train_y = np.copy(y)
        self.potentials = np.zeros_like(y, dtype=int)
        
        for _ in range(iterations):
            for i in range(self.train_X.shape[0]):
                if self.predict(X[i]) != y[i]:
                    self.potentials[i] += 1
                
        self.zero_indexes = np.where(self.potentials == 0)[0]
        self.nonzero_indexes = np.nonzero(self.potentials)
        self.train_X = self.train_X[self.nonzero_indexes]
        self.train_y = self.train_y[self.nonzero_indexes]
        self.potentials = self.potentials[self.nonzero_indexes]
    