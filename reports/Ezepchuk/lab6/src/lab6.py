import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 20*np.pi, 0.01)
y = 0.3 * np.cos(0.3 * x) + 0.07 * np.sin(0.3 * x)

window = 10
X, Y = [], []

for i in range(len(x) - window):
    X.append(y[i:i+window])
    Y.append(y[i+window])

X = np.array(X)
Y = np.array(Y).reshape(-1, 1)

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

class MultiRecurrentNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn1 = nn.RNN(input_size, hidden_size, batch_first=True)
        self.rnn2 = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(2)
        out1, _ = self.rnn1(x)
        out1 = self.sigmoid(out1)

        out2, _ = self.rnn2(out1)
        out2 = self.sigmoid(out2)

        out = self.fc(out2[:, -1, :])
        return out

model = MultiRecurrentNet(input_size=1, hidden_size=4)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 1000
losses = []

print("Начало обучения...\n")

for epoch in range(num_epochs):
    output = model(X_tensor)
    loss = criterion(output, Y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch + 1) % 50 == 0:
        print(f"Эпоха {epoch+1:3d}/{num_epochs} | Loss: {loss.item():.6f}")

print("\nОбучение завершено!\n")

with torch.no_grad():
    prediction = model(X_tensor).numpy()

mae = np.mean(np.abs(Y - prediction))
rmse = np.sqrt(np.mean((Y - prediction)**2))

print(f"Средняя абсолютная ошибка (MAE): {mae:.6f}")
print(f"Корень из среднеквадратичной ошибки (RMSE): {rmse:.6f}")

plt.style.use('seaborn-v0_8')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(x[window:], Y, label='Истинная функция', linewidth=1.2)
axes[0].plot(x[window:], prediction, label='Предсказание', linewidth=1)
axes[0].set_title("Функция и предсказание", fontsize=11)
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend(frameon=True)
axes[0].grid(alpha=0.3)

axes[1].plot(losses, linewidth=1)
axes[1].set_title("Ошибка обучения", fontsize=11)
axes[1].set_xlabel("Эпоха")
axes[1].set_ylabel("MSE")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
