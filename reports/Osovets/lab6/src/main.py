import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

# === 1. Генерация обучающих и тестовых данных ===
def target_function(x, a=0.3, b=0.1, c=0.06, d=0.1):
    return a * torch.cos(b * x) + c * torch.sin(d * x)

# Параметры варианта
a, b, c, d = 0.3, 0.1, 0.06, 0.1
input_size = 6
hidden_size = 2
num_samples = 200
train_ratio = 0.8

# Входы: 6 признаков — степени x
x = torch.linspace(0, 10, num_samples).reshape(-1, 1)
X = torch.cat([x ** i for i in range(1, input_size + 1)], dim=1)
y = target_function(x, a, b, c, d)

# Train/Test split
split_idx = int(num_samples * train_ratio)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print("Задание 3: Данные сгенерированы для функции y = a*cos(bx) + c*sin(dx)")
print(f"Размер обучающей выборки: {X_train.shape}, тестовой: {X_test.shape}")

# === 2. Архитектура РНС Элмана ===
class ElmanRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # RNN с tanh внутри (ограничение PyTorch)
        self.rnn = nn.RNN(input_dim, hidden_dim, nonlinearity="tanh", batch_first=True)
        self.sigmoid = nn.Sigmoid()   # добавляем сигмоид вручную
        self.fc = nn.Linear(hidden_dim, 1)  # линейный выход

    def forward(self, x):
        # РНС ожидает входы в формате [batch, seq, features]
        x = x.unsqueeze(1)  # добавляем размерность seq=1
        out, _ = self.rnn(x)
        out = self.sigmoid(out[:, -1, :])  # сигмоид на скрытом слое
        out = self.fc(out)                 # линейный выход
        return out

model = ElmanRNN(input_size, hidden_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# === 3. Обучение модели ===
losses = []
epochs = 500

for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

print("Обучение завершено. Минимальная ошибка:", min(losses))

# === 4. График ошибки по эпохам ===
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.title("График изменения ошибки по эпохам (РНС Элмана)")
plt.xlabel("Эпоха")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

# === 5. График прогнозируемой функции на обучающем участке ===
model.eval()
with torch.no_grad():
    y_train_pred = model(X_train)

plt.figure(figsize=(10, 4))
plt.plot(x[:split_idx], y_train, label="Эталон")
plt.plot(x[:split_idx], y_train_pred, label="Прогноз РНС Элмана")
plt.title("Прогнозируемая функция на обучающем участке")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# === 6. Таблица результатов обучения ===
train_table = pd.DataFrame({
    "Эталонное значение": y_train.squeeze().numpy(),
    "Полученное значение": y_train_pred.squeeze().numpy(),
    "Отклонение": (y_train_pred - y_train).squeeze().numpy()
})
print("\nРезультаты обучения (первые строки):")
print(train_table.head())

# === 7. Таблица результатов прогнозирования ===
with torch.no_grad():
    y_test_pred = model(X_test)
    test_table = pd.DataFrame({
        "Эталонное значение": y_test.squeeze().numpy(),
        "Полученное значение": y_test_pred.squeeze().numpy(),
        "Отклонение": (y_test_pred - y_test).squeeze().numpy()
    })
print("\nРезультаты прогнозирования (первые строки):")
print(test_table.head())