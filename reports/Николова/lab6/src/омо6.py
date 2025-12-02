import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

a, b, c, d = 0.2, 0.6, 0.05, 0.6

input_nodes = 10
hidden_nodes = 4
output_nodes = 1

def generate_function(x):
    return a * np.cos(b * x) + c * np.sin(d * x)

def create_dataset(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_weights_multirecurrent(input_size, hidden_size, output_size, context_size=1):
    np.random.seed(42)
    W1 = np.random.randn(input_size + context_size, hidden_size) * 0.1
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.1
    b2 = np.zeros((1, output_size))
    W_context = np.random.randn(output_size, context_size) * 0.1
    return W1, b1, W2, b2, W_context

def forward_propagation(X, W1, b1, W2, b2, W_context, context):
    X_with_context = np.hstack([X, context])
    hidden_input = np.dot(X_with_context, W1) + b1
    hidden_output = sigmoid(hidden_input)
    output = np.dot(hidden_output, W2) + b2  
    new_context = np.dot(output, W_context)
    return hidden_output, output, new_context

def backward_propagation(X, y, hidden_output, output, context,
                         W1, W2, W_context, b1, b2, learning_rate):
    m = y.shape[0]
    output_error = output - y.reshape(-1, 1)
    d_output = output_error / m

    dW2 = np.dot(hidden_output.T, d_output)
    db2 = np.sum(d_output, axis=0, keepdims=True)

    hidden_error = np.dot(d_output, W2.T)
    d_hidden = hidden_error * sigmoid_derivative(hidden_output)

    X_with_context = np.hstack([X, context])
    dW1 = np.dot(X_with_context.T, d_hidden)
    db1 = np.sum(d_hidden, axis=0, keepdims=True)

    dW_context = np.dot(output.T, d_hidden[:, -W_context.shape[1]:])

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W_context -= learning_rate * dW_context

    return W1, b1, W2, b2, W_context, np.mean(np.square(output_error))


def train_network(X_train, y_train, epochs=3000, learning_rate=0.01):
    W1, b1, W2, b2, W_context = initialize_weights_multirecurrent(input_nodes, hidden_nodes, output_nodes)

    errors = []
    context = np.zeros((X_train.shape[0], 1))

    for epoch in range(epochs):
        hidden_output, output, new_context = forward_propagation(
            X_train, W1, b1, W2, b2, W_context, context
        )

        W1, b1, W2, b2, W_context, error = backward_propagation(
            X_train, y_train, hidden_output, output, context,
            W1, W2, W_context, b1, b2, learning_rate
        )

        context = new_context
        errors.append(error)

        if epoch % 200 == 0:
            print(f"Эпоха {epoch}, Ошибка: {error:.6f}")

    return W1, b1, W2, b2, W_context, errors

def predict(X, W1, b1, W2, b2, W_context, sequence_length=1):
    predictions = []
    context = np.zeros((X.shape[0], 1))

    for _ in range(sequence_length):
        hidden_output, output, context = forward_propagation(
            X, W1, b1, W2, b2, W_context, context
        )
        predictions.append(output)

    return np.array(predictions).reshape(-1, 1)

if __name__ == "__main__":
    x_values = np.linspace(0, 100, 500)
    y_values = generate_function(x_values)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaled = scaler.fit_transform(y_values.reshape(-1, 1)).flatten()

    X, y = create_dataset(y_scaled, input_nodes)

    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")

    W1, b1, W2, b2, W_context, errors = train_network(
        X_train, y_train, epochs=5000, learning_rate=0.1
    )

    train_predictions = predict(X_train, W1, b1, W2, b2, W_context)
    test_predictions = predict(X_test, W1, b1, W2, b2, W_context)

    y_train_original = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    train_predictions_original = scaler.inverse_transform(train_predictions).flatten()

    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    test_predictions_original = scaler.inverse_transform(test_predictions).flatten()

    print("Результат: ")
    train_results = pd.DataFrame({
        'Эталонное значение': y_train_original,
        'Полученное значение': train_predictions_original,
        'Отклонение': np.abs(y_train_original - train_predictions_original)
    })
    print(train_results.head(10).round(6))

    print("Результат: ")
    test_results = pd.DataFrame({
        'Эталонное значение': y_test_original,
        'Полученное значение': test_predictions_original,
        'Отклонение': np.abs(y_test_original - test_predictions_original)
    })
    print(test_results.head(10).round(6))

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_train_original)), y_train_original, 'b-', label='Эталонные значения', linewidth=2)
    plt.plot(range(len(y_train_original)), train_predictions_original, 'r--', label='Прогноз РНС', linewidth=1.5)
    plt.title('Прогнозируемая функция на участке обучения')
    plt.xlabel('Временной шаг')
    plt.ylabel('Значение функции')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(errors)
    plt.title('Изменение ошибки в зависимости от итерации')
    plt.xlabel('Итерация')
    plt.ylabel('Среднеквадратичная ошибка')
    plt.grid(True)
    plt.show()

    train_mae = np.mean(np.abs(y_train_original - train_predictions_original))
    test_mae = np.mean(np.abs(y_test_original - test_predictions_original))
    