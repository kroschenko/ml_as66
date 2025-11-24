import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd


# 1. Загрузка датасета
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..','..', '..', '..', '..'))
file_path = os.path.join(project_root, 'Telco-Customer-Churn.csv')

data = pd.read_csv(file_path)

data = data.dropna()
data = data.drop(columns=["customerID"])

y = data["Churn"]
X = data.drop(columns=["Churn"])

for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

y = (y == "Yes").astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

input_dim = X_train.shape[1]


# 2. Определение архитектуры нейронной сети (без Dropout)
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(input_dim, 24)
        self.relu = nn.ReLU()
        self.output = nn.Linear(24, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x


# 3. Инициализация модели, функции потерь и оптимизатора
model = MLP(input_dim)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Цикл обучения
epochs = 50
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Эпоха [{epoch + 1}/{epochs}], Потери: {loss.item():.4f}")

# 5. Оценка модели
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test)
    y_pred_classes = (torch.sigmoid(y_pred_test) > 0.5).float()

accuracy = accuracy_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes)
print("\n=== Результаты без Dropout ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
print(classification_report(y_test, y_pred_classes))


# Эксперимент: добавляем Dropout 0.2


class MLP_Dropout(nn.Module):
    def __init__(self, input_dim):
        super(MLP_Dropout, self).__init__()
        self.hidden = nn.Linear(input_dim, 24)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(24, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


model_dropout = MLP_Dropout(input_dim)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model_dropout.parameters(), lr=0.001)

for epoch in range(epochs):
    model_dropout.train()
    y_pred = model_dropout(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Эпоха [{epoch + 1}/{epochs}] (Dropout), Потери: {loss.item():.4f}")

model_dropout.eval()
with torch.no_grad():
    y_pred_test = model_dropout(X_test)
    y_pred_classes = (torch.sigmoid(y_pred_test) > 0.5).float()

accuracy_d = accuracy_score(y_test, y_pred_classes)
f1_d = f1_score(y_test, y_pred_classes)
print("\n=== Результаты с Dropout(0.2) ===")
print(f"Accuracy: {accuracy_d:.4f}")
print(f"F1-score: {f1_d:.4f}")
print(classification_report(y_test, y_pred_classes))

print("\n=== Сравнение ===")
print(f"Без Dropout: Accuracy={accuracy:.4f}, F1={f1:.4f}")
print(f"С Dropout:   Accuracy={accuracy_d:.4f}, F1={f1_d:.4f}")
if f1_d > f1:
    print("✅ Dropout помог улучшить результат!")
else:
    print("❌ Dropout не улучшил результат.")
