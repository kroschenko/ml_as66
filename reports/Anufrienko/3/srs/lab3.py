import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Загрузка датасета
df = pd.read_csv("iris.csv")

target_col = "variety" 

X = df.drop(columns=[target_col])
y = df[target_col]

# 2. Разделение данных на train/test выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------------
# 3. Обучение трех моделей
# -----------------------------------------------------------

# --- k-NN с k=5 (базовое значение)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# --- Decision Tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# --- SVM
svm = SVC(kernel="rbf", random_state=42)
svm.fit(X_train, y_train)

# -----------------------------------------------------------
# 4. Подбор оптимального k
# -----------------------------------------------------------

k_values = range(1, 31)
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, preds))

best_k = k_values[accuracies.index(max(accuracies))]

print("Лучший k:", best_k)

# --- График зависимости точности от k ---
plt.plot(k_values, accuracies)
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Зависимость точности k-NN от k")
plt.grid(True)
plt.show()

# -----------------------------------------------------------
# 5. Оценка точности моделей
# -----------------------------------------------------------

knn_pred = knn.predict(X_test)
tree_pred = tree.predict(X_test)
svm_pred = svm.predict(X_test)

acc_knn = accuracy_score(y_test, knn_pred)
acc_tree = accuracy_score(y_test, tree_pred)
acc_svm = accuracy_score(y_test, svm_pred)

print("Точность k-NN (k=5):", acc_knn)
print("Точность Decision Tree:", acc_tree)
print("Точность SVM:", acc_svm)
print("Лучшая точность k-NN (подбор k):", max(accuracies))

