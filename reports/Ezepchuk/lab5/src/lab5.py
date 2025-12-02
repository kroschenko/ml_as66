import os
import numpy as np
import matplotlib.pyplot as plt

a, b, c, d = 0.3, 0.3, 0.07, 0.3
window_size = 10
hidden_size = 4
epochs = 1000
lr = 0.05
train_ratio = 0.7

OUT_DIR = "lab5_results"
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

def pretty_style():
    plt.style.use("seaborn-v0_8") 
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 10,
        "lines.linewidth": 2.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8"
    })

def main():
    pretty_style()

    x_vals, y_vals = generate_series(a, b, c, d, x_min=-50, x_max=50)

    X, Y = make_supervised_from_series(y_vals, window_size)
    split = int(train_ratio * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = Y[:split], Y[split:]

    W1, b1, W2, b2 = init_weights(window_size, hidden_size, output_size=1, seed=42)
    W1, b1, W2, b2, loss_history = train(X_train, y_train, W1, b1, W2, b2, epochs, lr)

    y_train_pred = predict(X_train, W1, b1, W2, b2)
    y_test_pred = predict(X_test, W1, b1, W2, b2)

    output_path = os.path.join(OUT_DIR, "all_predictions.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("===== TRAIN PREDICTIONS =====\n")
        for i, (true, pred) in enumerate(zip(y_train.flatten(), y_train_pred.flatten())):
            f.write(f"Точка {i:5d}: Истинное = {true:.6f} | Предсказание = {pred:.6f}\n")

        f.write("\n===== TEST PREDICTIONS =====\n")
        for i, (true, pred) in enumerate(zip(y_test.flatten(), y_test_pred.flatten()),
                                        start=len(y_train)):
            f.write(f"Точка {i:5d}: Истинное = {true:.6f} | Предсказание = {pred:.6f}\n")

    print(f"\nВсе предсказания сохранены в файл: {output_path}\n")

    def print_head_tail(name, true_vals, pred_vals, offset=0):
        total = len(true_vals)
        show_n = 10

        print(f"===== {name} (первые {show_n}) =====")
        for i in range(min(show_n, total)):
            print(f"Точка {i+offset:5d}: Истинное = {true_vals[i]:.6f} | Предсказание = {pred_vals[i]:.6f}")

        print(f"===== {name} (последние {show_n}) =====")
        for i in range(max(0, total - show_n), total):
            print(f"Точка {i+offset:5d}: Истинное = {true_vals[i]:.6f} | Предсказание = {pred_vals[i]:.6f}")


    print_head_tail("TRAIN", y_train.flatten(), y_train_pred.flatten(), offset=0)
    print_head_tail("TEST", y_test.flatten(), y_test_pred.flatten(), offset=len(y_train))


    plt.figure(figsize=(8, 4.5))
    epochs_range = np.arange(1, len(loss_history) + 1)
    plt.plot(epochs_range, loss_history, label="MSE (train)", linewidth=2)
    plt.fill_between(epochs_range, loss_history, np.max(loss_history), alpha=0.06)
    final_loss = loss_history[-1]
    plt.title("График изменения ошибки (MSE) по эпохам")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE")
    plt.grid(alpha=0.35)
    plt.legend()
    plt.annotate(f"Final MSE = {final_loss:.4e}", xy=(0.98, 0.95), xycoords="axes fraction",
                 ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8"))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"))
    plt.show()

    plt.figure(figsize=(8, 4.5))
    plt.plot(x_vals, y_vals, lw=2.2, label="Эталонная функция")
    plt.title("Эталонная функция")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "true_function_dense.png"))
    plt.show()

    plt.figure(figsize=(8, 4.5))
    idx = np.arange(len(y_vals))

    plt.plot(idx, y_vals, lw=2, label="Эталонная функция", zorder=1)

    train_start = window_size
    train_end = window_size + len(y_train_pred)
    test_start = train_end
    test_end = train_end + len(y_test_pred)

    plt.axvspan(train_start, train_end - 1, alpha=0.06, label="Train region")
    plt.axvspan(test_start, test_end - 1, alpha=0.04, label="Test region", color="C1")

    plt.plot(range(train_start, train_end),
             y_train_pred.flatten(), "--", marker="o", markersize=4, label="Прогноз (train)", zorder=3)
    plt.plot(range(test_start, test_end),
             y_test_pred.flatten(), "--", marker="s", markersize=4, label="Прогноз (test)", zorder=3)

    
    plt.title("Прогнозируемая функция — участки обучения и теста")
    plt.xlabel("Индекс точки")
    plt.ylabel("Значение y")
    plt.grid(alpha=0.25)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "prediction_vs_true.png"))
    plt.show()

if __name__ == "__main__":
    main()
