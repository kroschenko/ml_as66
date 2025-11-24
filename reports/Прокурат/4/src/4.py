import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report


df = pd.read_csv("glass.csv")

X = df.drop("Type", axis=1).values
y = df["Type"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

input_size = X_train.shape[1]
num_classes = len(le.classes_)


# 10x2
class MLP_2Hidden(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

# 20x1
class MLP_1Hidden(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


def train_model(model, X_train, y_train, epochs=1000, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)

    acc = accuracy_score(y_test, predicted)
    f1 = f1_score(y_test, predicted, average='weighted')
    print("Accuracy:", round(acc, 3))
    print("F1-score:", round(f1, 3), "\n")

    y_true_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(predicted)
    print(classification_report(y_true_labels, y_pred_labels, digits=3, zero_division=0))


print("Модель с двумя скрытыми слоями по 10 нейронов:")
model_2hidden = MLP_2Hidden(input_size, num_classes)
train_model(model_2hidden, X_train, y_train)
evaluate_model(model_2hidden, X_test, y_test)

print("\nМодель с одним скрытым слоем на 20 нейронов:")
model_1hidden = MLP_1Hidden(input_size, num_classes)
train_model(model_1hidden, X_train, y_train)
evaluate_model(model_1hidden, X_test, y_test)
