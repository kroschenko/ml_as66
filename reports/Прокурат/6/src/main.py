import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd


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


class MultiRecurrentRNN(nn.Module):
    def __init__(self, n_inputs, n_hidden):
        super(MultiRecurrentRNN, self).__init__()
        self.hidden_layer = nn.Linear(n_inputs + n_hidden, n_hidden)
        self.output_layer = nn.Linear(n_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h_prev):
        x_combined = torch.cat((x, h_prev), dim=1)
        h = self.sigmoid(self.hidden_layer(x_combined))
        y = self.output_layer(h)
        return y, h


def evaluate_model(model, X):
    preds = []
    h_prev = torch.zeros((1, n_hidden))
    for i in range(len(X)):
        pred, h_prev = model(X[i:i+1], h_prev)
        preds.append(pred.item())
    return np.array(preds)


learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
best_lr, best_loss = None, float('inf')

print("Подбор оптимального α:")
for lr in learning_rates:
    model = MultiRecurrentRNN(n_inputs, n_hidden)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    h_prev = torch.zeros((X_train.shape[0], n_hidden))

    for epoch in range(500):
        pred, h_prev = model(X_train, h_prev)
        loss = criterion(pred, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        h_prev = h_prev.detach()

    test_pred = evaluate_model(model, X_test)
    mse = np.mean((test_pred - Y_test.flatten().numpy()) ** 2)
    print(f"  α={lr:.3f} -> MSE={mse:.6f}")

    if mse < best_loss:
        best_loss = mse
        best_lr = lr

print(f"\nОптимальное значение α: {best_lr:.3f}, минимальная ошибка: {best_loss:.6f}\n")


model = MultiRecurrentRNN(n_inputs, n_hidden)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=best_lr)

losses = []
h_prev = torch.zeros((X_train.shape[0], n_hidden))

for epoch in range(n_epochs):
    pred, h_prev = model(X_train, h_prev)
    loss = criterion(pred, Y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    h_prev = h_prev.detach()
    losses.append(loss.item())


# 4.	График прогнозируемой функции на участке обучения
train_pred = pred.detach().numpy()
plt.figure(figsize=(10,5))
plt.plot(Y_train.numpy(), label="Эталон")
plt.plot(train_pred, label="Прогноз")
plt.title("График прогнозируемой функции на участке обучения")
plt.legend()
plt.grid()
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
plt.figure(figsize=(8,4))
plt.plot(losses)
plt.title("График изменения ошибки")
plt.xlabel("Эпоха")
plt.ylabel("MSE")
plt.grid()
plt.show()


# 6.	Результаты прогнозирования
test_pred = evaluate_model(model, X_test)
test_results = pd.DataFrame({
    "Эталонное значение": Y_test.numpy().flatten(),
    "Полученное значение": test_pred.flatten(),
})
test_results["Отклонение"] = test_results["Полученное значение"] - test_results["Эталонное значение"]

print("\nРезультаты прогнозирования (первые 10 значений):")
print(test_results.head(10))
