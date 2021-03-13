import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from random import seed

a = np.loadtxt('dane3.txt')

x = a[:, [0]]  # input  # t
y = a[:, [1]]  # output # p

P1, p1_test, T1, t1_test = train_test_split(x, y, test_size=0.3, random_state=42)

seed(1)  # jesli tego nie bedzie to bierze seeda w oparciu o aktualna sekunde

# definicja sieci

S1 = len(P1)
W1 = np.random.rand(S1, 1) - 0.5
B1 = np.random.rand(S1, 1) - 0.5
W2 = np.random.rand(1, S1) - 0.5
B2 = np.random.rand(1, 1) - 0.5

lr = 0.0001  # stała jak szybko idziemy w kierunku zmiany

P = np.transpose(P1)  # transpozycja

iter = 1
print("Start loop")

while iter < 1000:
    s = W1 @ P + B1 @ np.ones(P.shape)
    A1 = np.arctan(s)

    A2 = (W2 @ A1) + B2

    E2 = T1 - A2  # blad dla wszystkich probek uczacych
    E1 = np.transpose(W2) * E2

    dw2 = lr * E2 @ np.transpose(A1)
    db2 = lr * E2 @ np.transpose(np.ones(E2.shape))
    dw1 = lr * (1-(1 + s*s)) * E1 @ np.transpose(P)

    db1 = lr * (1-(1 + s*s)) * E1 @ np.transpose(np.ones(P.shape))

    W2 = W2 + dw2
    B2 = B2 + db2

    W1 = W1 + dw1
    B1 = B1 + db1
    iter += 1
    if np.mod(iter, 20) == 0:
        # print(iter)
        plt.clf()
        plt.plot(P1, T1, 'ro')
        plt.plot(P1, A2, 'b*', alpha=0.15)
        # time.sleep(0.5)
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
