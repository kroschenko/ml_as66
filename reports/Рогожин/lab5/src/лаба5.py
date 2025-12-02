import numpy as np
import matplotlib.pyplot as plt

a, b, c, d = 0.2, 0.4, 0.09, 0.4

def func(x):
    return a * np.cos(b * x) + c * np.sin(d * x)

x = np.linspace(0, 20, 400)
y = func(x)

window = 6
X, Y = [], []

for i in range(len(y) - window):
    X.append(y[i:i + window])
    Y.append(y[i + window])

X = np.array(X)
Y = np.array(Y)

train_size = 300
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

np.random.seed(0)
hidden = 2

W1 = np.random.randn(window, hidden) * 0.5
b1 = np.zeros(hidden)
W2 = np.random.randn(hidden) * 0.5
b2 = 0.0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

lr = 0.05
epochs = 800
errors = []

for epoch in range(epochs):
    z1 = X_train @ W1 + b1
    a1 = sigmoid(z1)
    y_pred = a1 @ W2 + b2

    err = y_pred - Y_train
    mse = np.mean(err ** 2)
    errors.append(mse)

    dW2 = a1.T @ err * (2 / len(X_train))
    db2 = np.mean(err) * 2

    da1 = err[:, None] * W2
    dz1 = da1 * (a1 * (1 - a1))

    dW1 = X_train.T @ dz1 * (2 / len(X_train))
    db1 = np.mean(dz1, axis=0) * 2

    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

def mlp_predict(X):
    return sigmoid(X @ W1 + b1) @ W2 + b2

train_pred = mlp_predict(X_train)
test_pred = mlp_predict(X_test)

plt.figure(figsize=(8, 4))
plt.plot(Y_train, label="Эталон")
plt.plot(train_pred, label="Прогноз")
plt.title("Прогноз на обучающем участке")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(errors)
plt.title("Ошибка обучения (MSE)")
plt.xlabel("Итерация")
plt.ylabel("Ошибка")
plt.grid()
plt.show()

def print_table(name, Y_true, Y_pred, limit=20):
    print("\n-----", name, "-----")
    print("Эталон\t\tПрогноз\t\tОтклонение")
    for t, p in list(zip(Y_true, Y_pred))[:limit]:
        print(f"{t:.6f}\t{p:.6f}\t{t - p:.6f}")

print_table("ОБУЧЕНИЕ", Y_train, train_pred)
print_table("ПРОГНОЗ", Y_test, test_pred)
