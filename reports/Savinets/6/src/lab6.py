import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

# Параметры варианта 11
a = 0.3
b = 0.5
c = 0.05
d = 0.5
n_inputs = 8
n_neurons = 3

# Генерация данных
def generate_function(x):
    return a * np.sin(b * x) + c * np.cos(d * x)

# Генерация последовательности данных
np.random.seed(42)
x = np.linspace(0, 20, 1000)
y = generate_function(x)

# Подготовка данных для RNN
def create_rnn_dataset(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps)])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

# Создание окон данных
X, y_target = create_rnn_dataset(y, n_inputs)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y_target, test_size=0.2, random_state=42, shuffle=False
)

# Преобразование формы данных для RNN (samples, time_steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"Форма X_train: {X_train.shape}")
print(f"Форма y_train: {y_train.shape}")
print(f"Форма X_test: {X_test.shape}")
print(f"Форма y_test: {y_test.shape}")

# Построение модели RNN Джордана
model = Sequential([
    SimpleRNN(n_neurons, activation='sigmoid', input_shape=(n_inputs, 1)),
    Dense(1, activation='linear')
])

# Компиляция модели
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

print("Архитектура модели:")
model.summary()

# Обучение модели
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

# Прогнозирование
y_train_pred = model.predict(X_train).flatten()
y_test_pred = model.predict(X_test).flatten()

# Расчет метрик
train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

# Вывод результатов обучения (первые 10 значений)
print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ (первые 10)")
print("="*50)
print("Эталонное значение | Полученное значение | Отклонение")
train_results = []
for i in range(min(10, len(y_train))):
    true_val = y_train[i]
    pred_val = y_train_pred[i]
    deviation = true_val - pred_val
    train_results.append((true_val, pred_val, deviation))
    print(f"{true_val:16.6f} {pred_val:20.6f} {deviation:13.6f}")

# Вывод результатов прогнозирования (первые 10 значений)
print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ (первые 10)")
print("="*50)
print("Эталонное значение | Полученное значение | Отклонение")
test_results = []
for i in range(min(10, len(y_test))):
    true_val = y_test[i]
    pred_val = y_test_pred[i]
    deviation = true_val - pred_val
    test_results.append((true_val, pred_val, deviation))
    print(f"{true_val:16.6f} {pred_val:20.6f} {deviation:13.6f}")

# Итоговые метрики
print("\n" + "="*50)
print("ИТОГОВЫЕ МЕТРИКИ")
print("="*50)
print(f"Train → MSE: {train_mse:.8f}, MAE: {train_mae:.8f}")
print(f"Test  → MSE: {test_mse:.8f}, MAE: {test_mae:.8f}")

# Сравнение с ЛР5
print("\n" + "="*50)
print("СРАВНЕНИЕ С ЛАБОРАТОРНОЙ РАБОТОЙ №5")
print("="*50)
print("ЛР5 (оптимальное a=0.15): Test MSE = 0.00000081")
print(f"ЛР6 (RNN Джордана):      Test MSE = {test_mse:.8f}")

if test_mse < 0.00000081:
    print("✓ RNN показала лучшую точность по сравнению с ЛР5")
else:
    print("✗ RNN показала худшую точность по сравнению с ЛР5")

# График 1: Прогнозируемая функция на участке обучения
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
# Используем часть данных для лучшей визуализации
plot_range = min(100, len(y_train))
plt.plot(range(plot_range), y_train[:plot_range], 'b-', label='Эталонные значения', linewidth=2)
plt.plot(range(plot_range), y_train_pred[:plot_range], 'r--', label='Прогноз RNN', linewidth=2)
plt.title('Результаты обучения RNN')
plt.xlabel('Временной шаг')
plt.ylabel('Значение функции')
plt.legend()
plt.grid(True)

# График 2: Результаты прогнозирования
plt.subplot(2, 2, 2)
plot_range_test = min(50, len(y_test))
plt.plot(range(plot_range_test), y_test[:plot_range_test], 'b-', label='Эталонные значения', linewidth=2)
plt.plot(range(plot_range_test), y_test_pred[:plot_range_test], 'r--', label='Прогноз RNN', linewidth=2)
plt.title('Результаты прогнозирования RNN')
plt.xlabel('Временной шаг')
plt.ylabel('Значение функции')
plt.legend()
plt.grid(True)

# График 3: Изменение ошибки по эпохам
plt.subplot(2, 2, 3)
plt.plot(history.history['loss'], 'b-', label='Ошибка обучения')
plt.plot(history.history['val_loss'], 'r-', label='Ошибка валидации')
plt.title('Изменение ошибки по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

# График 4: Сравнение эталонных и прогнозируемых значений
plt.subplot(2, 2, 4)
plt.scatter(y_train, y_train_pred, alpha=0.5, label='Обучение')
plt.scatter(y_test, y_test_pred, alpha=0.5, label='Тестирование')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'k--', linewidth=2)
plt.title('Сравнение эталонных и прогнозируемых значений')
plt.xlabel('Эталонные значения')
plt.ylabel('Прогнозируемые значения')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Выводы
print("\n" + "="*50)
print("ВЫВОДЫ")
print("="*50)
print("1. Архитектура сети: RNN Джордана с 8 входами, 3 нейронами в скрытом слое")
print("2. Функции активации: сигмоида (скрытый слой), линейная (выходной слой)")
print("3. Оптимизатор: Adam с learning rate = 0.01")
print("4. Обучение: 200 эпох, batch size = 16")

if test_mse < 0.00000081:
    print("5. Точность: RNN превзошла результаты ЛР5 по тестовой MSE")
    print("6. Устойчивость: RNN лучше обобщает временные зависимости")
else:
    print("5. Точность: RNN уступила по точности модели из ЛР5")
    print("6. Причина: возможно, требуется больше данных или настройка гиперпараметров")

print("7. RNN эффективна для временных рядов благодаря памяти о предыдущих состояниях")