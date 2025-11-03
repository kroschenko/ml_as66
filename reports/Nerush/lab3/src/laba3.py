import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

# 1. Загрузка и предобработка данных
df = pd.read_csv("E:/Projects/ml_as66/reports/Nerush/lab3/src/adult.csv", na_values='?')
print(f"Загружено строк: {len(df)}")

# Удаление строк с пропущенными значениями
df = df.dropna()
print(f"После удаления пропусков осталось строк: {len(df)}")

# Преобразование целевой переменной
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
print("Целевая переменная преобразована в бинарный формат.")

# Кодирование категориальных признаков
categorical_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"После кодирования категориальных признаков: {df.shape[1]} признаков.")

# 2. Разделение на обучающую и тестовую выборки
X = df.drop('income', axis=1)
y = df['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Данные разделены на обучающую и тестовую выборки.")

# 3. Обучение моделей
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_knn = knn.predict(X_test)
precision_knn = precision_score(y_test, y_knn)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
y_tree = tree.predict(X_test)
precision_tree = precision_score(y_test, y_tree)

svm = LinearSVC(max_iter=10000)
svm.fit(X_train, y_train)
y_svm = svm.predict(X_test)
precision_svm = precision_score(y_test, y_svm)

# 4. Сравнение моделей
print("\nPrecision для класса '>50K'")
print(f"k-NN (k=5): {precision_knn:.3f}")
print(f"Decision Tree: {precision_tree:.3f}")
print(f"LinearSVC: {precision_svm:.3f}")

# 5. Исследование влияния параметра k
k_values = range(1, 21)
precisions = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    p = precision_score(y_test, y_pred)
    precisions.append(p)

plt.figure(figsize=(8, 6))
plt.plot(k_values, precisions, marker='o', linestyle='-', color='blue')
plt.title("Precision для класса '>50K' при разных k")
plt.xlabel("Количество соседей (k)")
plt.ylabel("Precision")
plt.grid(True)
plt.tight_layout()
plt.show()