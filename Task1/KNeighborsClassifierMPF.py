import math
import numpy as np
from scipy.spatial import distance


def epanechnikov(r):
    return 3.0 / 4 * (1 - r ** 2) if abs(r) <= 1.0 else 0.0


def quadratic(r):
    return 15.0 / 16 * (1 - r ** 2) if abs(r) <= 1.0 else 0.0


def triangular(r):
    return 1 - abs(r) if abs(r) <= 1.0 else 0.0


def gauss(r):
    return math.pow(2 * math.pi, -1.0 / 2) * math.exp(-1 / 2 * (r ** 2))


def rect(r):
    return 1 / 2 if abs(r) <= 1.0 else 0.0


class KNeighborsClassifierMPF:
    kernels = {
        "epanechnikov": epanechnikov,
        "quadratic": quadratic,
        "triangular": triangular,
        "gauss": gauss}

    def __init__(
            self,
            metricPower=2,
            kernel="gauss"
    ):
        self.p = metricPower
        self.kernel = self.kernels[kernel]
        self.metric = self.__getMetric()

    def __calculateH(self, X):
        maxValue = 0
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                val = self.metric(X[i], X[j])
                if val > maxValue:
                    maxValue = val
        return maxValue

    def __getMetric(self):
        return lambda u, v: distance.minkowski(u, v, p=self.p, w=None)

    def __w(self, x1, x2, h, gamma):
        return gamma * self.kernel(self.metric(x1, x2) / h)

    def predict(self, X, y, weight, xu, H):
        classWeight = np.zeros(np.max(y) - np.min(y) + 1)
        for i in range(len(weight)):
            classWeight[y[i]] += self.__w(xu, X[i], H, weight[i])
        return np.argmax(classWeight)

    def weightFit(self, X, y, iterNumber=10):
        weight = np.zeros(len(y))
        H = self.__calculateH(X)
        for j in range(iterNumber):
            for i in range(len(weight)):
                if self.predict(X, y, weight, X[i], 2 * H) != y[i]:
                    weight[i] += 1
        return weight

    def score(self, X, y, weight, x_test, y_test):
        TPN = 0
        H = self.__calculateH(X)
        for i in range(len(y_test)):
            prediction = self.predict(X, y, weight, x_test[i], 2 * H)
            if prediction == y_test[i]:
                TPN += 1
        return TPN / len(y_test)
