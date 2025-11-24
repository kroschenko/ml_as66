import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix

torch.manual_seed(42)
np.random.seed(42)

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "kddcup.data_10_percent_corrected")

df = pd.read_csv(dataset_path, header=None)
df.columns = [f"col_{i}" for i in range(df.shape[1])]
df["target"] = (df["col_41"] != "normal.").astype(int)

X = df.drop(columns=["target", "col_41"])
y = df["target"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

X_train_raw, X_test_raw, y_train_np, y_test_np = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = preprocess.fit_transform(X_train_raw)
X_test = preprocess.transform(X_test_raw)

if hasattr(X_train, "toarray"):
    X_train = X_train.toarray()
    X_test = X_test.toarray()

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_np.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test_np.values, dtype=torch.float32).view(-1, 1)

input_dim = X_train_tensor.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor = X_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

class MLP64(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class MLP16(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def train_and_evaluate(ModelClass, input_dim, n_epochs=10, lr=0.001):
    model = ModelClass(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    start_time = time.time()

    for epoch in range(n_epochs):
        model.train()
        logits = model(X_train_tensor)
        loss = criterion(logits, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} Loss: {loss.item():.4f}")

    train_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_tensor)
        probs_test = torch.sigmoid(logits_test)
        preds = (probs_test > 0.5).float()

    y_true = y_test_tensor.cpu().numpy().ravel()
    y_pred = preds.cpu().numpy().ravel()
    y_prob = probs_test.cpu().numpy().ravel()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    plt.figure(figsize=(7, 4))
    plt.plot(loss_history)
    plt.title(f"Loss — {ModelClass.__name__}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], '--', color="gray")
    plt.title(f"ROC — {ModelClass.__name__}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid(True)
    plt.show()

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix — {ModelClass.__name__}")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
    plt.show()

    return acc, f1, train_time


print("\n==== 64 neurons ====")
acc64, f164, time64 = train_and_evaluate(MLP64, input_dim)
print(f"Accuracy={acc64:.4f}, F1={f164:.4f}, Time={time64:.2f}")

print("\n==== 16 neurons ====")
acc16, f116, time16 = train_and_evaluate(MLP16, input_dim)
print(f"Accuracy={acc16:.4f}, F1={f116:.4f}, Time={time16:.2f}")

print("\n==== Comparison ====")
print(f"ΔAccuracy={acc16 - acc64:.4f}")
print(f"ΔTime={time16 - time64:.2f}")

# ======== GRAPH: MODEL COMPARISON ========

models = ["MLP64", "MLP16"]
acc_values = [acc64, acc16]
f1_values = [f164, f116]
time_values = [time64, time16]

x = np.arange(len(models))
w = 0.25

plt.figure(figsize=(9, 5))
plt.bar(x - w, acc_values, width=w, label="Accuracy")
plt.bar(x, f1_values, width=w, label="F1-score")
plt.bar(x + w, time_values, width=w, label="Time")

plt.xticks(x, models)
plt.title("Сравнение моделей (64 vs 16 нейронов)")
plt.ylabel("Значение")
plt.legend()
plt.grid(True)
plt.show()
