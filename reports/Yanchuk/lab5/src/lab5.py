import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a, b, c, d = 0.2, 0.2, 0.06, 0.2
window_size = 8
hidden_size = 3
epochs = 2500
lr = 0.05
train_ratio = 0.7

OUT_DIR = "lab5_variant2_results"
os.makedirs(OUT_DIR, exist_ok=True)

def sigmoid(x):
    x_clip = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x_clip))

def sigmoid_deriv_from_activation(a_sigmoid):
    return a_sigmoid * (1.0 - a_sigmoid)

def generate_series(a, b, c, d, x_min=-50, x_max=50, step=0.01):
    x_vals = np.arange(x_min, x_max + step, step)
    y_vals = a * np.cos(b * x_vals) + c * np.sin(d * x_vals)
    return x_vals, y_vals

def make_supervised_from_series(y_vals, window):
    X, Y = [], []
    for i in range(len(y_vals) - window):
        X.append(y_vals[i:i + window])
        Y.append(y_vals[i + window])
    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64).reshape(-1, 1)
    return X, Y

def init_weights(input_size, hidden_size, output_size=1, seed=42):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0.0, np.sqrt(1.0 / input_size), size=(input_size, hidden_size))
    b1 = np.zeros((1, hidden_size), dtype=np.float64)

    W2 = rng.normal(0.0, np.sqrt(1.0 / hidden_size), size=(hidden_size, output_size))
    b2 = np.zeros((1, output_size), dtype=np.float64)
    return W1, b1, W2, b2

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    y_pred = z2
    cache = (X, z1, a1, z2)
    return y_pred, cache

def predict(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    return a1 @ W2 + b2

def compute_mse(y_pred, y_true):
    err = y_pred - y_true
    return np.mean(err ** 2), err

def backward_and_update(W1, b1, W2, b2, cache, err, lr):
    X, z1, a1, z2 = cache
    N = X.shape[0]

    grad_out = 2.0 * err / N

    dW2 = a1.T @ grad_out
    db2 = np.sum(grad_out, axis=0, keepdims=True)

    da1 = grad_out @ W2.T
    dz1 = da1 * sigmoid_deriv_from_activation(a1)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    return W1, b1, W2, b2

def train(X_train, y_train, W1, b1, W2, b2, epochs, lr):
    loss_history = []
    for epoch in range(1, epochs + 1):
        y_pred, cache = forward(X_train, W1, b1, W2, b2)
        loss, err = compute_mse(y_pred, y_train)
        loss_history.append(loss)

        W1, b1, W2, b2 = backward_and_update(W1, b1, W2, b2, cache, err, lr)

        if epoch % 250 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | MSE={loss:.8f}")

    return W1, b1, W2, b2, loss_history

def main():
    x_vals, y_vals = generate_series(a, b, c, d, x_min=-50, x_max=50)

    X, Y = make_supervised_from_series(y_vals, window_size)
    split = int(train_ratio * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = Y[:split], Y[split:]

    W1, b1, W2, b2 = init_weights(window_size, hidden_size, output_size=1, seed=42)
    W1, b1, W2, b2, loss_history = train(X_train, y_train, W1, b1, W2, b2, epochs, lr)

    y_train_pred = predict(X_train, W1, b1, W2, b2)
    y_test_pred = predict(X_test, W1, b1, W2, b2)

    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, color="tab:blue", label="MSE")
    plt.title("График изменения ошибки (MSE) по эпохам")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=150)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(x_vals, y_vals, lw=2, color="tab:green", label="Эталонная функция")
    plt.title("Эталонная функция (плотная дискретизация)")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "true_function_dense.png"), dpi=150)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(y_vals)), y_vals, lw=2, label="Эталонная функция")
    plt.plot(range(window_size, window_size + len(y_train_pred)),
             y_train_pred.flatten(), "--", label="Прогноз (train)")
    plt.plot(range(window_size + len(y_train_pred),
                   window_size + len(y_train_pred) + len(y_test_pred)),
             y_test_pred.flatten(), "--", label="Прогноз (test)")
    plt.title("Прогнозируемая функция на участке обучения и теста")
    plt.xlabel("Индекс точки")
    plt.ylabel("Значение y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "prediction_vs_true.png"), dpi=150)
    plt.show()

    print(f"\nФайлы сохранены в: {OUT_DIR}")
    print(" - train_results.csv, test_results.csv")
    print(" - loss_curve.png, true_function_dense.png, prediction_vs_true.png, prediction_vs_true_long.png")


if __name__ == "__main__":
    main()


