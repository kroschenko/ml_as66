import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    recall_score
)

# ------------------------------
# Загружаем данные
# ------------------------------
data = pd.read_csv('breast_cancer.csv')

# Убираем лишнюю колонку, которая полностью пустая
if 'Unnamed: 32' in data.columns:
    data.drop('Unnamed: 32', axis=1, inplace=True)

# Преобразуем метки: B -> 0, M -> 1
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

# Разделяем признаки и целевую переменную
X = data.drop(['id', 'diagnosis'], axis=1)
y = data['diagnosis']

# ------------------------------
# Обработка пропусков
# ------------------------------
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Масштабируем данные
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# ------------------------------
# Разделяем на train/test
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# Подбор лучшего k для KNN
# ------------------------------
recall_scores = []
k_range = range(1, 21)

for k in k_range:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train, y_train)
    y_pred = knn_k.predict(X_test)
    recall = recall_score(y_test, y_pred, pos_label=1)
    recall_scores.append(recall)

best_k = k_range[np.argmax(recall_scores)]
print(f'Лучшее k для KNN по recall (malignant): {best_k}')

# ------------------------------
# Обучаем все модели
# ------------------------------
knn = KNeighborsClassifier(n_neighbors=best_k)
dt = DecisionTreeClassifier(random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)

knn.fit(X_train, y_train)
dt.fit(X_train, y_train)
svm.fit(X_train, y_train)

models = {
    'k-NN': knn,
    'Decision Tree': dt,
    'SVM': svm
}

# ------------------------------
# Оценка моделей
# ------------------------------
recall_scores_for_malignant = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Benign', 'Malignant'])
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, pos_label=1)

    print(f"\n{name}:\nМатрица ошибок:\n{cm}\n")
    print(f"Отчет классификации:\n{report}")
    print(f"Precision (malignant): {precision[0]:.4f}")
    print(f"Recall (malignant): {recall[0]:.4f}")
    print(f"F1-score (malignant): {f1[0]:.4f}")

    recall_scores_for_malignant[name] = recall_score(y_test, y_pred, pos_label=1)

# ------------------------------
# Лучшая модель по recall для malignant
# ------------------------------
best_model_name = max(recall_scores_for_malignant, key=recall_scores_for_malignant.get)
best_recall = recall_scores_for_malignant[best_model_name]

print(f"\nМодель, минимизирующая ложноотрицательные для malignant: {best_model_name}")
print(f"Recall (malignant) = {best_recall:.4f}")

best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)
cm_best = confusion_matrix(y_test, y_pred_best)
print(f"\nМатрица ошибок для лучшей модели ({best_model_name}):\n{cm_best}")
