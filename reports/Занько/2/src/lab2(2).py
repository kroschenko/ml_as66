import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

data = pd.read_csv("bank.csv", sep=";")

df = data.copy()
for col in df.select_dtypes(include="object").columns:
    if col != "y":
        df[col] = LabelEncoder().fit_transform(df[col])

df["y"] = df["y"].map({"yes": 1, "no": 0})

X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Качество модели:")
print("Accuracy =", round(accuracy, 2))
print("Precision (yes) =", round(precision, 2))
print("Recall (yes) =", round(recall, 2))

cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["yes", "no"])
disp.plot(cmap="Blues")
plt.title("Матрица ошибок")
plt.show()