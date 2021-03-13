import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from random import seed

a = np.loadtxt('dane3.txt')

x = a[:, [0]]  # input  # T
y = a[:, [1]]  # output # P

# P1, p1_test, T1, t1_test = train_test_split(x, y, test_size=0.25, random_state=1)

seed(1)



S1 = len(x)
W1 = np.random.rand(S1, 1) - 0.5
B1 = np.random.rand(S1, 1) - 0.5
W2 = np.random.rand(1, S1) - 0.5
B2 = np.random.rand(1, 1) - 0.5

lr = 0.0001  # stała jak szybko idziemy w kierunku zmiany

P = np.transpose(x)

iter = 1
print("Start loop")

while iter < 2000:
    # s = W1 @ P + B1 @ np.ones(P.shape)
    X = W1 * P + B1 * np.ones(P.shape)
    A1 = np.fmax(X, 0)

    A2 = (W2 @ A1) + B2

    E2 = y - A2
    E1 = np.transpose(W2) * E2

    dw2 = lr * E2 @ np.transpose(A1)
    db2 = lr * E2 @ np.transpose(np.ones(E2.shape))

    # dw1 = lr * (-np.sin(s)) * E1 @ np.transpose(P)
    # db1 = lr * (-np.sin(s)) * E1 @ np.transpose(np.ones(P.shape))

    # dw1 = lr * (1-(1 + s*s)) * E1 @ np.transpose(P)
    # db1 = lr * (1-(1 + s*s)) * E1 @ np.transpose(np.ones(P.shape))

    dw1 = lr * np.divide(np.exp(X), (np.exp(X + 1))) * E1 @ np.transpose(P)
    db1 = lr * np.divide(np.exp(X), (np.exp(X + 1))) * E1 @ np.transpose(np.ones(P.shape))

    W2 = W2 + dw2
    B2 = B2 + db2

    W1 = W1 + dw1
    B1 = B1 + db1
    iter += 1
    if np.mod(iter, 10) == 0:
        # print(iter)
        plt.clf()
        plt.plot(x, y, 'ro')
        plt.plot(x, A2, alpha=0.35)
        time.sleep(0.2)
        plt.show()

print("End loop\n")

print("W1")
print(W1)
print("--------------------------")
print("B1")
print(B1)
print("--------------------------")
print("W2")
print(W2)
print("--------------------------")
print("B2")
print(B2)
