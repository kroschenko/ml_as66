import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


a, b, c, d = 0.1, 0.3, 0.08, 0.3

n_inputs = 10
n_hidden = 4
n_epochs = 2000


def func(x):
    return a * np.cos(b * x) + c * np.sin(d * x)


x = np.linspace(0, 30, 500)
y = func(x)

X, Y = [], []
for i in range(len(y) - n_inputs):
    X.append(y[i:i + n_inputs])
    Y.append(y[i + n_inputs])

X, Y = np.array(X), np.array(Y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

torch.manual_seed(42)
X_train, Y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
X_test, Y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_inputs, n_hidden)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        return self.out(x)


learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
best_lr = None
best_loss = float("inf")

criterion = nn.MSELoss()

print("Подбор оптимального α:")
for lr in learning_rates:
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(700):
        optimizer.zero_grad()
        loss = criterion(model(X_train), Y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        mse = criterion(model(X_test), Y_test).item()

    print(f"  α={lr:.3f}\tMSE={mse:.6f}")

    if mse < best_loss:
        best_loss = mse
        best_lr = lr

print(f"\nОптимальное значение α: {best_lr:.3f}, минимальная ошибка: {best_loss:.6f}\n")

model = Net()
optimizer = optim.Adam(model.parameters(), lr=best_lr)
losses = []

for epoch in range(n_epochs):
    optimizer.zero_grad()
    loss = criterion(model(X_train), Y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())


# 4.    График прогнозируемой функции на участке обучения
with torch.no_grad():
    train_pred = model(X_train).numpy()

plt.figure(figsize=(10, 4))
plt.plot(Y_train.numpy(), label="Эталон")
plt.plot(train_pred, label="Прогноз")
plt.title("График прогнозируемой функции на участке обучения")
plt.grid(True)
plt.legend()
plt.show()


# 5.	Результаты обучения
# таблица
train_results = pd.DataFrame({
    "Эталонное значение": Y_train.numpy().flatten(),
    "Полученное значение": train_pred.flatten(),
})
train_results["Отклонение"] = train_results["Полученное значение"] - train_results["Эталонное значение"]

print("\nРезультаты обучения (первые 10 значений):")
print(train_results.head(10))


# график
plt.figure(figsize=(7, 4))
plt.plot(losses)
plt.xlabel("Эпоха")
plt.ylabel("MSE")
plt.title("График изменения ошибки в зависимости от итерации")
plt.grid(True)
plt.show()


# 6.	Результаты прогнозирования
with torch.no_grad():
    test_pred = model(X_test).numpy()

test_results = pd.DataFrame( { "Эталонное значение": Y_test.numpy().flatten(), "Полученное значение": test_pred.flatten(), } )
test_results["Отклонение"] = test_results["Полученное значение"] - test_results["Эталонное значение"]

print("\nРезультаты прогнозирования (первые 10 значений):")
print(test_results.head(10))

