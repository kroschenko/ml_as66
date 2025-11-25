import numpy as np
import matplotlib.pyplot as plt

a = 0.2
b = 0.6
c = 0.05
d = 0.6

INPUTS = 10
HIDDEN = 4

lr = 0.01
epochs = 1000
clip_grad = 5.0

def f(t):
    return a * np.cos(b * t) + c * np.sin(d * t)


N = 500
t = np.linspace(0, 50, N)
y = f(t)

X = []
Y = []
for i in range(N - INPUTS):
    X.append(y[i:i + INPUTS])  #
    Y.append(y[i + INPUTS])
X = np.array(X)
Y = np.array(Y).reshape(-1, 1)

samples = X.shape[0]

train_size = int(0.8 * samples)
test_size = samples - train_size

X_train = X[:train_size]
Y_train = Y[:train_size]
X_test = X[train_size:]
Y_test = Y[train_size:]

rng = np.random.RandomState(123)

Wx = rng.normal(scale=0.5, size=(HIDDEN, 1))
Wh = rng.normal(scale=0.5, size=(HIDDEN, HIDDEN))
bh = np.zeros((HIDDEN, 1))
Who = rng.normal(scale=0.5, size=(1, HIDDEN))
bo = np.zeros((1, 1))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(y):
    return y * (1 - y)


def forward_window(x_window):
    h_prev = np.zeros((HIDDEN, 1))
    h_list = []
    z_list = []
    y_list = []

    for t_step in range(len(x_window)):
        x_t = np.array([[x_window[t_step]]])
        z_t = Wx @ x_t + Wh @ h_prev + bh
        h_t = sigmoid(z_t)
        y_t = (Who @ h_t) + bo

        h_list.append(h_t)
        z_list.append(z_t)
        y_list.append(y_t)
        h_prev = h_t

    return {
        'z': z_list,
        'h': h_list,
        'y': y_list
    }

loss_history = []

for ep in range(epochs):
    epoch_loss = 0.0

    for i in range(samples):
        x_win = X[i]
        target = np.array([[Y[i, 0]]])

        hist = forward_window(x_win)
        y_pred = hist['y'][-1]

        err = y_pred - target
        loss = 0.5 * (err ** 2)
        epoch_loss += loss.item()

        h_list = hist['h']
        z_list = hist['z']

        dWx = np.zeros_like(Wx)
        dWh = np.zeros_like(Wh)
        dbh = np.zeros_like(bh)
        dWho = np.zeros_like(Who)
        dbo = np.zeros_like(bo)

        dY = (y_pred - target)
        dWho += dY @ h_list[-1].T
        dbo += dY

        dh_next = np.zeros((HIDDEN, 1))
        delta = (Who.T @ dY) * dsigmoid(h_list[-1])
        dh = delta + dh_next

        for t_step in reversed(range(INPUTS)):
            h_t = h_list[t_step]

            if t_step == INPUTS - 1:
                delta_t = (Who.T @ dY) * dsigmoid(h_t)
            else:
                delta_t = dh * dsigmoid(h_t)

            x_t = np.array([[x_win[t_step]]])

            dWx += delta_t @ x_t.T
            if t_step > 0:
                dWh += delta_t @ h_list[t_step - 1].T
            dbh += delta_t

            dh = Wh.T @ delta_t

        for g in [dWx, dWh, dbh, dWho, dbo]:
            np.clip(g, -clip_grad, clip_grad, out=g)

        Wx -= lr * dWx
        Wh -= lr * dWh
        bh -= lr * dbh
        Who -= lr * dWho
        bo -= lr * dbo

    mse = epoch_loss / samples
    loss_history.append(mse)

    if (ep + 1) % 100 == 0 or ep == 0:
        print(f"Epoch {ep + 1} | MSE={mse:.6f}")

y_pred_train = np.zeros((train_size,))
for i in range(train_size):
    hist = forward_window(X_train[i])
    y_pred_train[i] = hist['y'][-1].item()

y_pred_test = np.zeros((test_size,))
for i in range(test_size):
    hist = forward_window(X_test[i])
    y_pred_test[i] = hist['y'][-1].item()

y_pred_all = np.zeros((samples,))
for i in range(samples):
    hist = forward_window(X[i])
    y_pred_all[i] = hist['y'][-1].item()

print("\n" + "=" * 50)
print("ОБУЧАЮЩАЯ ВЫБОРКА (первые 10)")
print("=" * 50)
print("    Эталонное   Полученное   Отклонение")
for i in range(10):
    error = y_pred_train[i] - Y_train[i, 0]
    print(f"{i}  {Y_train[i, 0]: .6f}  {y_pred_train[i]: .6f}    {error: .6f}")

print("\n" + "=" * 50)
print("ТЕСТОВАЯ ВЫБОРКА (первые 10)")
print("=" * 50)
print("    Эталонное   Полученное   Отклонение")
for i in range(10):
    error = y_pred_test[i] - Y_test[i, 0]
    print(f"{i}  {Y_test[i, 0]: .6f}  {y_pred_test[i]: .6f}    {error: .6f}")

forecast_steps = 50
start_seq = list(y[-INPUTS:])
forecast = []
cur = start_seq.copy()

for step in range(forecast_steps):
    hist = forward_window(np.array(cur))
    pred = hist['y'][-1].item()
    forecast.append(pred)
    cur = cur[1:] + [pred]

plt.figure(figsize=(12, 5))
plt.plot(y, label="Истинная функция (y)")
plt.plot(np.arange(INPUTS, N), y_pred_all, label="RNN прогноз (на обучении)", linewidth=2)
plt.title("Реальная функция и прогноз RNN (вариант 6, мультирекуррентная)")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(loss_history)
plt.title("Изменение ошибки (loss) по эпохам")
plt.xlabel("эпоха")
plt.ylabel("средний loss")
plt.grid()
plt.show()

print("\n" + "=" * 50)
print("ПРОГНОЗИРОВАНИЕ (50 шагов)")
print("=" * 50)
print("Шаг\tПрогноз")
for i, val in enumerate(forecast):
    print(f"{i + 1}\t{val:.6f}")

train_errors = y_pred_train - Y_train.flatten()
test_errors = y_pred_test - Y_test.flatten()

print("\n" + "=" * 50)
print("СТАТИСТИКА ОШИБОК")
print("=" * 50)
print(f"Обучающая выборка - MSE: {np.mean(train_errors ** 2):.6f}")
print(f"Тестовая выборка - MSE: {np.mean(test_errors ** 2):.6f}")