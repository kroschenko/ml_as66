import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

DATA_FILE = "heart_disease_uci.csv"
CM_IMAGE = "confusion_matrix.png"
TEST_PART = 0.2
SEED = 42

def main():
    # Проверка наличия файла
    if not os.path.isfile(DATA_FILE):
        print(f"Нет файла: {DATA_FILE}")
        sys.exit(1)

    # Загрузка данных
    data = pd.read_csv(DATA_FILE)
    print("Форма датасета:", data.shape)

    # Проверим столбец цели
    if "num" not in data.columns:
        print("Не найден целевой столбец 'num'.")
        sys.exit(1)

    # Целевой признак: есть ли заболевание (1) или нет (0)
    y = (data["num"] > 0).astype(int)
    X = data.drop(columns=["num"])

    # Обработка категориальных данных (если есть)
    categorical = X.select_dtypes(include="object").columns
    if len(categorical) > 0:
        print("Преобразуем категориальные:", list(categorical))
        X = pd.get_dummies(X, columns=categorical, drop_first=True)

    # Заполнение пропусков
    filler = SimpleImputer(strategy="mean")
    X = pd.DataFrame(filler.fit_transform(X), columns=X.columns)

    # Масштабирование признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделение выборок
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_PART, random_state=SEED, stratify=y
    )

    # Логистическая регрессия
    model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Оценка
    print("\n===== Метрики =====")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred, zero_division=0):.4f}")

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(CM_IMAGE, dpi=200)
    plt.close()

    print(f"\nМатрица ошибок сохранена как: {CM_IMAGE}")

if __name__ == "__main__":
    main()
