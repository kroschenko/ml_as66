import numpy as np
import matplotlib.pyplot as plt

a, b, c, d = 0.2, 0.4, 0.09, 0.4

def func(x):
    return a * np.cos(b * x) + c * np.sin(d * x)

x = np.linspace(0, 20, 400)
y = func(x)

window = 6
hidden = 2

X, Y = [], []
for i in range(len(y) - window):
    X.append(y[i:i+window])
    Y.append(y[i+window])
X = np.array(X)
Y = np.array(Y)

train_size = 300
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

np.random.seed(0)
Wx = np.random.randn(window, hidden) * 0.5
Wh = np.random.randn(hidden, hidden) * 0.5
b1 = np.zeros(hidden)

Wy = np.random.randn(hidden) * 0.5
b2 = 0.0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

lr = 0.05
epochs = 500
errors = []

for epoch in range(epochs):
    h_prev = np.zeros(hidden)
    y_preds = []
    hs = []

    for t in range(len(X_train)):
        x_t = X_train[t]
        h = sigmoid(x_t @ Wx + h_prev @ Wh + b1)
        y_pred = h @ Wy + b2
        y_preds.append(y_pred)
        hs.append(h)
        h_prev = h

    y_preds = np.array(y_preds)
    err = y_preds - Y_train
    mse = np.mean(err ** 2)
    errors.append(mse)

    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db1 = np.zeros_like(b1)
    dWy = np.zeros_like(Wy)
    db2 = 0.0
    dh_next = np.zeros(hidden)

    for t in reversed(range(len(X_train))):
        h = hs[t]
        h_prev = hs[t-1] if t > 0 else np.zeros(hidden)
        dy = err[t]

        dWy += h * dy * (2 / len(X_train))
        db2 += dy * 2 / len(X_train)

        dh = Wy * dy * 2 / len(X_train) + dh_next
        dz = dh * h * (1 - h)

        dWx += np.outer(X_train[t], dz)
        dWh += np.outer(h_prev, dz)
        db1 += dz

        dh_next = Wh @ dz

    Wx -= lr * dWx
    Wh -= lr * dWh
    b1 -= lr * db1
    Wy -= lr * dWy
    b2 -= lr * db2

def rnn_predict(X):
    h_prev = np.zeros(hidden)
    y_preds = []
    for t in range(len(X)):
        h = sigmoid(X[t] @ Wx + h_prev @ Wh + b1)
        y_pred = h @ Wy + b2
        y_preds.append(y_pred)
        h_prev = h
    return np.array(y_preds)

train_pred = rnn_predict(X_train)
test_pred = rnn_predict(X_test)

plt.figure(figsize=(8,4))
plt.plot(Y_train, label="Эталон")
plt.plot(train_pred, label="Прогноз")
plt.title("RNN Элмана - обучение")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(errors)
plt.title("Ошибка обучения (MSE)")
plt.xlabel("Эпоха")
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
