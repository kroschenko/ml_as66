import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv('breast_cancer.csv')
df = df.drop('Unnamed: 32', axis=1)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


class BreastCancerNN(nn.Module):
    def __init__(self, input_size, hidden_size=16):
        super(BreastCancerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def train_and_evaluate_model(hidden_size, model_name):
    print(f"=== Модель с {hidden_size} нейронами ===")

    model = BreastCancerNN(input_size=X_train_tensor.shape[1], hidden_size=hidden_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    train_losses = []

    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train_tensor)
        loss = criterion(y_pred, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        y_pred_proba = model(X_test_tensor)
        y_pred = (y_pred_proba > 0.5).float()
        y_pred_np = y_pred.numpy().flatten()
        y_test_np = y_test_tensor.numpy().flatten()

        accuracy = accuracy_score(y_test_np, y_pred_np)
        precision = precision_score(y_test_np, y_pred_np)
        recall = recall_score(y_test_np, y_pred_np)
        f1 = f1_score(y_test_np, y_pred_np)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(classification_report(y_test_np, y_pred_np, target_names=['Доброкачественная', 'Злокачественная']))

    return accuracy, precision, recall, f1, train_losses


acc_16, prec_16, rec_16, f1_16, losses_16 = train_and_evaluate_model(16, "16 нейронов")
acc_32, prec_32, rec_32, f1_32, losses_32 = train_and_evaluate_model(32, "32 нейрона")

print("\n" + "=" * 40)
print("ФИНАЛЬНЫЙ РЕЗУЛЬТАТ:")
print("=" * 40)
print(f"16 нейронов: P={prec_16:.3f} R={rec_16:.3f} F1={f1_16:.3f}")
print(f"32 нейрона:  P={prec_32:.3f} R={rec_32:.3f} F1={f1_32:.3f}")
print(f"Δ: P={prec_32 - prec_16:+.3f} R={rec_32 - rec_16:+.3f}")

if rec_32 > rec_16:
    print("✅ Recall вырос - меньше пропусков рака")
else:
    print("❌ Recall упал")