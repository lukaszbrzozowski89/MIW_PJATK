import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from cw3.plotka import plot_decision_regions


class Perceptron(object):

    def __init__(self, eta=0.1, n_iter=1000):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, x, y):
        self.w_ = np.zeros(1 + x.shape[1])

        for _ in range(self.n_iter):
            for xi, target in zip(x, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
        return self

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)


class Classifier:

    def __init__(self, ppn1, ppn2):
        self.ppn1 = ppn1
        self.ppn2 = ppn2

    def predict(self, x):
        return np.where(self.ppn1.predict(x) == 1, 0, np.where(self.ppn2.predict(x) == 1, 2, 1))


def main():
    iris = datasets.load_iris()
    x = iris.data[:, [2, 3]]
    y = iris.target
    print('y= ', y)

    x_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=1, stratify=y)

    y_train_03_subset = y_train.copy()
    y_train_01_subset = y_train.copy()
    x_train_01_subset = x_train.copy()

    y_train_01_subset[(y_train == 1) | (y_train == 2)] = -1
    y_train_01_subset[(y_train_01_subset == 0)] = 1

    y_train_03_subset[(y_train == 1) | (y_train == 0)] = -1
    y_train_03_subset[(y_train_03_subset == 2)] = 1

    print('y_train_01_subset = ', y_train_01_subset)
    print('y_train_03_subset = ', y_train_03_subset)

    ppn = Perceptron()
    ppn.fit(x_train_01_subset, y_train_01_subset)

    ppn2 = Perceptron()
    ppn2.fit(x_train_01_subset, y_train_03_subset)

    y_1_predict = ppn.predict(x_train)
    y_3_predict = ppn2.predict(x_train)

    accuracy_1 = accuracy_score(y_1_predict, y_train_01_subset)
    accuracy_3 = accuracy_score(y_3_predict, y_train_03_subset)

    if accuracy_1 > accuracy_3:
        y_results = np.where(y_1_predict == 1, 0, np.where(y_3_predict == 1, 2, 1))
    else:
        y_results = np.where(y_3_predict == 1, 2, np.where(y_1_predict == 1, 0, 1))
    print('Accuracy: ', accuracy_score(y_results, y_train), '%', sep="")

    classifier = Classifier(ppn, ppn2)
    plot_decision_regions(x, y, classifier=classifier)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
