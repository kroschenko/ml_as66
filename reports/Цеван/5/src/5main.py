import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

a = 0.1
b = 0.1
c = 0.05
d = 0.1
n_inputs = 6
n_hidden = 2

def target_function(x):
    return a * np.cos(b * x) + c * np.sin(d * x)

x = np.linspace(-100, 300, 4000)
y = target_function(x)

def create_dataset(data, n_inputs):
    X, Y = [], []
    for i in range(len(data) - n_inputs):
        X.append(data[i:i + n_inputs])
        Y.append(data[i + n_inputs])
    return np.array(X), np.array(Y)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x_scaled = scaler_x.fit_transform(x.reshape(-1, 1)).flatten()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

X, Y = create_dataset(y_scaled, n_inputs)

start_train = np.where(x >= 50)[0][0] - n_inputs
end_train = np.where(x <= 100)[0][-1] - n_inputs
start_test = np.where(x >= 100)[0][0] - n_inputs
end_test = np.where(x <= 150)[0][-1] - n_inputs

X_train, X_test = X[start_train:end_train], X[start_test:end_test]
Y_train, Y_test = Y[start_train:end_train], Y[start_test:end_test]

X_train = torch.FloatTensor(X_train)
Y_train = torch.FloatTensor(Y_train).reshape(-1, 1)
X_test = torch.FloatTensor(X_test)
Y_test = torch.FloatTensor(Y_test).reshape(-1, 1)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.output(x)
        return x

model = NeuralNetwork(n_inputs, n_hidden, 1)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_losses = []
test_losses = []
epochs = 500

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    train_loss = criterion(outputs, Y_train)
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, Y_test)

    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())

    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{epochs}], Train Loss: {train_loss.item():.6f}, Test Loss: {test_loss.item():.6f}')

model.eval()
with torch.no_grad():
    train_predictions = model(X_train)
    test_predictions = model(X_test)

Y_train_actual = scaler_y.inverse_transform(Y_train.numpy())
train_predictions_actual = scaler_y.inverse_transform(train_predictions.numpy())
Y_test_actual = scaler_y.inverse_transform(Y_test.numpy())
test_predictions_actual = scaler_y.inverse_transform(test_predictions.numpy())

plt.figure(figsize=(20, 12))

plt.subplot(2, 2, 1)
x_train_plot = x[start_train + n_inputs:end_train + n_inputs]
plt.plot(x_train_plot, Y_train_actual, 'b-', label='True', linewidth=1.5, alpha=0.8)
plt.plot(x_train_plot, train_predictions_actual, 'r--', label='Prediction', linewidth=1.5)
plt.title('Training interval (50–100)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(train_losses, 'g-', label='Train loss', linewidth=1.5)
plt.plot(test_losses, 'r-', label='Test loss', linewidth=1.5)
plt.title('Loss curves')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.yscale('log')

plt.subplot(2, 2, 3)
x_test_plot = x[start_test + n_inputs:end_test + n_inputs]
plt.plot(x_test_plot, Y_test_actual, 'b-', label='True', linewidth=1.5, alpha=0.8)
plt.plot(x_test_plot, test_predictions_actual, 'r--', label='Prediction', linewidth=1.5)
plt.title('Test interval (100–150)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.scatter(Y_test_actual, test_predictions_actual, s=20, alpha=0.6)
y_min = min(Y_test_actual.min(), test_predictions_actual.min())
y_max = max(Y_test_actual.max(), test_predictions_actual.max())
plt.plot([y_min, y_max], [y_min, y_max], 'k--')
plt.title('True vs Predicted')
plt.xlabel('True')
plt.ylabel('Predicted')
plt.grid(True)

plt.tight_layout(pad=3.0)
plt.show()

train_results = pd.DataFrame({
    'True': Y_train_actual.flatten(),
    'Predicted': train_predictions_actual.flatten(),
    'Error': (Y_train_actual.flatten() - train_predictions_actual.flatten())
})

test_results = pd.DataFrame({
    'True': Y_test_actual.flatten(),
    'Predicted': test_predictions_actual.flatten(),
    'Error': (Y_test_actual.flatten() - test_predictions_actual.flatten())
})

print(train_results.head(10))
print(test_results.head(10))
