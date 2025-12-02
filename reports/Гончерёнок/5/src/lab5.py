import numpy as np
import matplotlib.pyplot as plt

a, b, c, d = 0.2, 0.2, 0.06, 0.2
n_inputs = 8
n_hidden = 12
learning_rate = 0.05
epochs = 5000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def generate_data(n=1000, step=0.1):
    x = np.arange(0, n * step, step)
    y = a * np.cos(d * x) + c * np.sin(d * x)
    return x, y


def prepare_data(x, y, n_inputs):
    X, Y = [], []
    for i in range(len(y) - n_inputs):
        X.append(y[i:i + n_inputs])
        Y.append(y[i + n_inputs])
    return np.array(X), np.array(Y)


x, y = generate_data()
X, Y = prepare_data(x, y, n_inputs)

X_mean, X_std = X.mean(), X.std()
Y_mean, Y_std = Y.mean(), Y.std()

X_normalized = (X - X_mean) / X_std
Y_normalized = (Y - Y_mean) / Y_std

split = int(0.8 * len(X_normalized))
X_train, Y_train = X_normalized[:split], Y_normalized[:split].reshape(-1, 1)
X_test, Y_test = X_normalized[split:], Y_normalized[split:].reshape(-1, 1)

np.random.seed(42)
W1 = np.random.randn(n_inputs, n_hidden) * np.sqrt(2.0 / n_inputs)
b1 = np.zeros((1, n_hidden))
W2 = np.random.randn(n_hidden, 1) * np.sqrt(2.0 / n_hidden)
b2 = np.zeros((1, 1))

loss_history = []

for epoch in range(epochs):
    Z1 = np.dot(X_train, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = Z2

    loss = np.mean((A2 - Y_train) ** 2)
    loss_history.append(loss)

    dZ2 = 2 * (A2 - Y_train) / len(Y_train)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = np.dot(X_train.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    if epoch == 0:
        vW1, vW2, vb1, vb2 = dW1, dW2, db1, db2
    else:
        beta = 0.9
        vW1 = beta * vW1 + (1 - beta) * dW1
        vW2 = beta * vW2 + (1 - beta) * dW2
        vb1 = beta * vb1 + (1 - beta) * db1
        vb2 = beta * vb2 + (1 - beta) * db2

    W1 -= learning_rate * vW1
    b1 -= learning_rate * vb1
    W2 -= learning_rate * vW2
    b2 -= learning_rate * vb2

    if epoch % 500 == 0:
        print(f"Эпоха {epoch}, Ошибка: {loss:.6f}")


def predict(X):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    return Z2


Y_train_pred_normalized = predict(X_train)
Y_test_pred_normalized = predict(X_test)

Y_train_pred = Y_train_pred_normalized * Y_std + Y_mean
Y_train_actual = Y_train * Y_std + Y_mean
Y_test_pred = Y_test_pred_normalized * Y_std + Y_mean
Y_test_actual = Y_test * Y_std + Y_mean

train_errors = Y_train_actual - Y_train_pred
test_errors = Y_test_actual - Y_test_pred

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(Y_train_actual[:200], label='Истинные значения', linewidth=2)
plt.plot(Y_train_pred[:200], label='Прогноз', linestyle='dashed', linewidth=1.5)
plt.title('Прогноз на обучающей выборке (первые 200 точек)')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.legend()
plt.grid()

plt.subplot(2, 3, 2)
plt.plot(loss_history)
plt.title('Изменение ошибки в процессе обучения')
plt.xlabel('Эпоха')
plt.ylabel('MSE (нормализованные данные)')
plt.grid()
plt.yscale('log')

plt.subplot(2, 3, 3)
plt.plot(Y_test_actual[:100], label='Истинные значения', linewidth=2)
plt.plot(Y_test_pred[:100], label='Прогноз', linestyle='dashed', linewidth=1.5)
plt.title('Прогноз на тестовой выборке (первые 100 точек)')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.legend()
plt.grid()

plt.subplot(2, 3, 4)
plt.plot(train_errors[:100])
plt.title('Ошибки на обучающей выборке')
plt.xlabel('Индекс')
plt.ylabel('Отклонение')
plt.grid()

plt.subplot(2, 3, 5)
plt.plot(test_errors[:100])
plt.title('Ошибки на тестовой выборке')
plt.xlabel('Индекс')
plt.ylabel('Отклонение')
plt.grid()

plt.subplot(2, 3, 6)
x_original = x[n_inputs:len(Y_train_actual) + n_inputs]
plt.plot(x_original[:300], Y_train_actual[:300], label='Истинные', alpha=0.7)
plt.plot(x_original[:300], Y_train_pred[:300], label='Прогноз', linestyle='--', alpha=0.9)
plt.title('Сравнение на отрезке x')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ (первые 10 значений):")
print("Эталон    | Прогноз    | Отклонение")
print("-" * 50)
for i in range(10):
    print(f"{Y_train_actual[i, 0]:10.6f} | {Y_train_pred[i, 0]:10.6f} | {train_errors[i, 0]:12.6f}")

print("\nСТАТИСТИКА ОШИБОК:")
print(f"Средняя абсолютная ошибка на обучении: {np.mean(np.abs(train_errors)):.8f}")
print(f"Среднеквадратичная ошибка на обучении: {np.sqrt(np.mean(train_errors ** 2)):.8f}")
print(f"Максимальная ошибка на обучении: {np.max(np.abs(train_errors)):.8f}")
print(f"Средняя абсолютная ошибка на тесте: {np.mean(np.abs(test_errors)):.8f}")
print(f"Среднеквадратичная ошибка на тесте: {np.sqrt(np.mean(test_errors ** 2)):.8f}")
print(f"Максимальная ошибка на тесте: {np.max(np.abs(test_errors)):.8f}")