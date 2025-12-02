import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

file_path = r"D:\у--ба\3 kurs\омо\4\iris.csv"

df = pd.read_csv(file_path)

# пусть последний столбец — это метка класса
X = df.iloc[:, :-1].values
y_raw = df.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y_raw)  # 3 класса: 0, 1, 2

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#тензоры PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

class MLP_1Hidden(nn.Module):
    def __init__(self):
        super(MLP_1Hidden, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )

    def forward(self, x):
        return self.model(x)

class MLP_2Hidden(nn.Module):
    def __init__(self):
        super(MLP_2Hidden, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 3)
        )

    def forward(self, x):
        return self.model(x)

model = MLP_2Hidden()#нужную модель выбрать MLP_1Hidden()/MLP_2Hidden() и запустить.

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100 # если плохая обучаемость то повысить на 200-300
for epoch in range(n_epochs):
    model.train()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

model.eval()#Оценка
with torch.no_grad():
    y_test_logits = model(X_test_tensor)
    y_test_pred = torch.argmax(y_test_logits, dim=1)

acc = accuracy_score(y_test_tensor, y_test_pred)
f1 = f1_score(y_test_tensor, y_test_pred, average='macro')

print(f"\nAccuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")