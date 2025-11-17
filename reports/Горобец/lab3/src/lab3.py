import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

df = pd.read_csv("C:/Users/Anton/Downloads/winequality-white.csv", sep=";")
df["Target"] = (df["quality"] >= 7).astype(int)  # 1 — хорошее, 0 — обычное
X = df.drop(["quality", "Target"], axis=1)
y = df["Target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

best_k = None
best_score = 0
f1_scores = []

# Перебираем значения k от 1 до 20
for k in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred)
    f1_scores.append(score)

    if score > best_score:
        best_score = score
        best_k = k
print(f"\nЛучшее значение k: {best_k}, F1-score: {best_score:.4f}")

models = {
    "k-NN": KNeighborsClassifier(n_neighbors=best_k),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel="rbf", random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"{name} F1-score: {f1:.4f}")
