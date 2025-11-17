
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

a, b, c, d = 0.2, 0.2, 0.06, 0.2
window_size = 8
hidden_size = 3
epochs = 1000
lr = 0.01

x_vals = np.arange(-10, 10.1, 0.1)
y_vals = a*np.cos(b*x_vals) + c*np.sin(d*x_vals)

X, Y = [], []
for i in range(len(y_vals)-window_size):
    X.append(y_vals[i:i+window_size])
    Y.append(y_vals[i+window_size])
X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32).reshape(-1,1)

split = int(0.7*len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = Y[:split], Y[split:]

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_test_t  = torch.tensor(X_test)
y_test_t  = torch.tensor(y_test)

class JordanRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc_in = nn.Linear(input_size + 1, hidden_size)
        self.act   = nn.Sigmoid()
        self.fc_out= nn.Linear(hidden_size, 1)

    def forward(self, X, teacher_forcing=True, y_true=None):
        N = X.shape[0]
        preds = []
        context = torch.zeros(N, 1, dtype=X.dtype, device=X.device)

        x_in = torch.cat([X, context], dim=1)
        h    = self.act(self.fc_in(x_in))
        y    = self.fc_out(h)
        preds.append(y)

        if teacher_forcing and (y_true is not None):
            context = y_true
        else:
            context = y.detach()

        return preds[0]

model = JordanRNN(window_size, hidden_size)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

loss_history = []
for epoch in range(1, epochs+1):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_t, teacher_forcing=True, y_true=y_train_t)
    loss = criterion(y_pred, y_train_t)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | MSE={loss.item():.6f}")

model.eval()
with torch.no_grad():
    y_train_pred = model(X_train_t, teacher_forcing=True, y_true=y_train_t).cpu().numpy()
    y_test_pred  = model(X_test_t, teacher_forcing=False).cpu().numpy()

train_df = pd.DataFrame({
    "Эталонное": y_train.flatten(),
    "Полученное": y_train_pred.flatten(),
    "Отклонение": (y_train_pred.flatten() - y_train.flatten())
})
test_df = pd.DataFrame({
    "Эталонное": y_test.flatten(),
    "Полученное": y_test_pred.flatten(),
    "Отклонение": (y_test_pred.flatten() - y_test.flatten())
})

print("\n=== Обучающая (первые 10) ===\n", train_df.head(10))
print("\n=== Тестовая (первые 10) ===\n", test_df.head(10))

plt.figure(figsize=(8,4))
plt.plot(loss_history, label="MSE")
plt.title("Ошибка по эпохам (PyTorch, Jordan)")
plt.xlabel("Эпоха")
plt.ylabel("MSE")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(range(len(y_vals)), y_vals, label="Эталонная функция", lw=2)
plt.plot(range(window_size, window_size+len(y_train_pred)), y_train_pred, '--', label="Прогноз (train)")
plt.plot(range(window_size+len(y_train_pred), window_size+len(y_train_pred)+len(y_test_pred)), y_test_pred, '--', label="Прогноз (test)")
plt.legend()
plt.title("Прогнозируемая функция vs Эталон")
plt.show()
