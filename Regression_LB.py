import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from cw3.plotka import plot_decision_regions


class LogisticRegressionGD(object):
    def __init__(self, eta=0.1, n_iter=3000, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.4, scale=0.1, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)

        return self

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


class Classifier2:
    def __init__(self, clas1, clas2):
        self.clas1 = clas1
        self.clas2 = clas2

    def predict(self, x):
        return np.where(self.clas1.predict(x) == 0, 0, np.where(self.clas2.predict(x) == 1, 2, 1))


def main():
    iris = datasets.load_iris()
    x = iris.data[:, [2, 3]]
    y = iris.target
    # print('x= ', x)
    print('y= ', y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=1, stratify=y)
    # w regresji logarytmicznej wyjście przyjmuje wartości 0 lub 1 (prawdopodobieństwa)

    y_train_01_subset = y_train.copy()
    y_train_02_subset = y_train.copy()
    x_train_01_subset = x_train.copy()
    x_train_02_subset = x_train.copy()

    y_train_01_subset[(y_train == 0)] = 0
    y_train_01_subset[(y_train_01_subset == 2)] = 1

    y_train_02_subset[(y_train == 1)] = 0
    y_train_02_subset[(y_train_02_subset == 2)] = 1

    print('y_train_01_subset = ', y_train_01_subset)
    print('y_train_02_subset = ', y_train_02_subset)

    lrgd = LogisticRegressionGD()
    lrgd.fit(x_train_01_subset, y_train_01_subset)
    lrgd2 = LogisticRegressionGD()
    lrgd2.fit(x_train_02_subset, y_train_02_subset)

    y_1_predict = lrgd.predict(x_train)
    y_2_predict = lrgd2.predict(x_train)

    accuracy_1 = accuracy_score(y_1_predict, y_train_01_subset)
    accuracy_3 = accuracy_score(y_2_predict, y_train_02_subset)

    if accuracy_1 > accuracy_3:
        y_results = np.where(y_1_predict == 0, 0, np.where(y_2_predict == 1, 2, 1))
    else:
        y_results = np.where(y_2_predict == 1, 2, np.where(y_1_predict == 1, 0, 1))
    print('Accuracy: ', accuracy_score(y_results, y_train), '%', sep="")

    classifier2 = Classifier2(lrgd, lrgd2)
    plot_decision_regions(x, y, classifier=classifier2)
    method(x, lrgd, lrgd2)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()

def method(x, r1, r2):
    wynik1 = r1.activation(r1.net_input(x))
    wynik2 = r2.activation(r2.net_input(x))
    print(wynik1)
    print(wynik2)


if __name__ == '__main__':
    main()
