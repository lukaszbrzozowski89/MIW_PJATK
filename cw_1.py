import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

a = np.loadtxt('cw1/dane3.txt')

x = a[:, [0]]  # INPUT
y = a[:, [1]]  # OUTPUT

x_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

liniowy = np.hstack([x_train, np.ones(x_train.shape)])  # model: y = ax + b
wektor_linia = np.linalg.pinv(liniowy) @ y_train  # @ mnozenie macierzy
print("------------------")
print("Model 1")
print(wektor_linia)
e_linia = sum((y_train - (wektor_linia[0] * x_train + wektor_linia[1])) ** 2) / len(x_train)
print("------------------")

print("E Model1- train")
print(e_linia)
print("------------------")
e_linia_test = sum((y_test - (wektor_linia[0] * X_test + wektor_linia[1])) ** 2) / len(X_test)
print("E Model1 -test")
print(e_linia_test)
print()
print("------------------\n")

print("Model 2")
model_2 = np.hstack([-np.cos(x_train), np.ones(x_train.shape)])  # model: y = -cos(Ax) + B
wektor_2 = np.linalg.pinv(model_2) @ y_train
print(wektor_2)
print("------------------")

e_cosinus = sum((y_train - (-np.cos(x_train * np.pi/2) * wektor_2[0] + wektor_2[1])) ** 2) / len(x_train)
print("E Model2 train")
print(e_cosinus)
print("------------------")

e_cosinus = sum((y_test - (-np.cos(X_test * np.pi/2) * wektor_2[0] + wektor_2[1])) ** 2) / len(X_test)
print("E Model2 testowe")
print(e_cosinus)

plt.plot(X_test, y_test, 'g^')
plt.plot(x, wektor_linia[0] * x + wektor_linia[1])  # liniowa
plt.plot(x_train, y_train, 'ro')

plt.plot(x, -np.cos(x * np.pi/2) * wektor_2[0] + wektor_2[1])  # trygonometryczna
plt.show()
