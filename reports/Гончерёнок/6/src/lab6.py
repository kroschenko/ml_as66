import numpy as np
import matplotlib.pyplot as plt

a = 0.2
b = 0.2
c = 0.06
d = 0.2
num_inputs = 8
hidden_neurons = 3
alpha_values = [0.001, 0.005, 0.01, 0.05, 0.1]

def generate_data(a, b, c, d, t_start=0, t_end=30, step=0.1):
    t = np.arange(t_start, t_end + step, step)
    y = c * np.sin(d * t) + a * np.cos(b * t)
    return t, y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def linear(x):
    return x

def create_sequences(data, seq_length):
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        Y.append(data[i + seq_length])
    return np.array(X), np.array(Y)

def initialize_weights(input_size, hidden_size, output_size):
    W_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
    W_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
    W_context_hidden = np.random.randn(hidden_size, hidden_size) * 0.1
    return W_input_hidden, W_hidden_output, W_context_hidden

def forward_pass(X, W_ih, W_ho, W_ch, context):
    hidden_input = np.dot(X, W_ih) + np.dot(context, W_ch)
    hidden_output = sigmoid(hidden_input)
    output = linear(np.dot(hidden_output, W_ho))
    return hidden_output, output

def train_jordan(X_train, Y_train, hidden_neurons, alpha, epochs=2000):
    input_size = X_train.shape[1]
    output_size = 1
    W_ih, W_ho, W_ch = initialize_weights(input_size, hidden_neurons, output_size)

    errors = []
    context = np.zeros((1, hidden_neurons))

    for epoch in range(epochs):
        epoch_error = 0
        for i in range(len(X_train)):
            x = X_train[i].reshape(1, -1)
            y_true = Y_train[i]

            hidden, y_pred = forward_pass(x, W_ih, W_ho, W_ch, context)

            error = y_true - y_pred.item()
            epoch_error += error ** 2

            delta_output = error
            delta_hidden = delta_output * W_ho.T * sigmoid_derivative(hidden)

            W_ho += alpha * hidden.T * delta_output
            W_ih += alpha * x.T * delta_hidden
            W_ch += alpha * context.T * delta_hidden

            context = hidden.copy()

        avg_error = epoch_error / len(X_train)
        errors.append(avg_error)
        if epoch % 500 == 0:
            print(f"Эпоха {epoch}, ошибка: {avg_error:.6f}")

    return W_ih, W_ho, W_ch, errors

def predict_jordan(X, W_ih, W_ho, W_ch):
    predictions = []
    context = np.zeros((1, hidden_neurons))
    for i in range(len(X)):
        x = X[i].reshape(1, -1)
        hidden, y_pred = forward_pass(x, W_ih, W_ho, W_ch, context)
        predictions.append(y_pred.item())
        context = hidden.copy()
    return np.array(predictions)

t, y = generate_data(a, b, c, d, step=0.1)
seq_length = num_inputs
X, Y = create_sequences(y, seq_length)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

print(f"Всего точек: {len(t)}")
print(f"Обучающих последовательностей: {len(X_train)}")
print(f"Тестовых последовательностей: {len(X_test)}")
print(f"Диапазон y: min={y.min():.3f}, max={y.max():.3f}")

best_alpha = None
best_error = float('inf')
best_weights = None
best_errors = []

for alpha in alpha_values:
    print(f"\nОбучение с alpha = {alpha}")
    W_ih, W_ho, W_ch, errors = train_jordan(X_train, Y_train, hidden_neurons, alpha, epochs=2000)
    predictions = predict_jordan(X_test, W_ih, W_ho, W_ch)
    test_error = np.mean((predictions - Y_test) ** 2)
    print(f"Ошибка на тесте: {test_error:.6f}")

    if test_error < best_error:
        best_error = test_error
        best_alpha = alpha
        best_weights = (W_ih, W_ho, W_ch)
        best_errors = errors

print(f"\nЛучший alpha: {best_alpha}, ошибка на тесте: {best_error:.6f}")

W_ih, W_ho, W_ch = best_weights
predictions_train = predict_jordan(X_train, W_ih, W_ho, W_ch)
predictions_test = predict_jordan(X_test, W_ih, W_ho, W_ch)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(t[:split + seq_length], y[:split + seq_length], label='Исходный ряд', color='blue')
axes[0, 0].plot(t[seq_length:split + seq_length], predictions_train, label='Прогноз (обучение)', linestyle='--',
                color='orange')
axes[0, 0].set_title(f'Обучение, alpha={best_alpha}')
axes[0, 0].set_xlabel('t')
axes[0, 0].set_ylabel('y')
axes[0, 0].legend()
axes[0, 0].grid()

axes[0, 1].plot(best_errors, color='red')
axes[0, 1].set_title('Ошибка обучения')
axes[0, 1].set_xlabel('Эпоха')
axes[0, 1].set_ylabel('MSE')
axes[0, 1].grid()

axes[1, 0].plot(t[split + seq_length:], y[split + seq_length:], label='Исходный ряд', color='blue')
axes[1, 0].plot(t[split + seq_length:], predictions_test, label='Прогноз (тест)', linestyle='--', color='green')
axes[1, 0].set_title('Прогнозирование на тесте')
axes[1, 0].set_xlabel('t')
axes[1, 0].set_ylabel('y')
axes[1, 0].legend()
axes[1, 0].grid()

axes[1, 1].axis('off')
table_data = []
for i in range(min(8, len(Y_test))):
    table_data.append([f"{t[split + seq_length + i]:.1f}",
                       f"{Y_test[i]:.3f}",
                       f"{predictions_test[i]:.3f}",
                       f"{abs(Y_test[i] - predictions_test[i]):.3f}"])

table = axes[1, 1].table(cellText=table_data,
                         colLabels=['t', 'Эталон', 'Прогноз', 'Отклонение'],
                         loc='center',
                         cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
axes[1, 1].set_title('Таблица отклонений (первые 8 тестовых)')

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ (первые 5):")
print("-" * 70)
print(f"{'t':>6} {'Эталон':>10} {'Прогноз':>10} {'Отклонение':>12}")
print("-" * 70)
for i in range(5):
    t_val = t[seq_length + i]
    print(
        f"{t_val:6.1f} {Y_train[i]:10.3f} {predictions_train[i]:10.3f} {abs(Y_train[i] - predictions_train[i]):12.3f}")

print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ (первые 8 тестовых):")
print("-" * 70)
print(f"{'t':>6} {'Эталон':>10} {'Прогноз':>10} {'Отклонение':>12}")
print("-" * 70)
for i in range(8):
    t_val = t[split + seq_length + i]
    print(f"{t_val:6.1f} {Y_test[i]:10.3f} {predictions_test[i]:10.3f} {abs(Y_test[i] - predictions_test[i]):12.3f}")