import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


CSV_PATH = "heart_disease_uci.csv"
OUT_PNG = "confusion_matrix.png"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Файл не найден: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print("Загружен датасет:", df.shape)
    print("Колонки:", list(df.columns))

    # Целевой столбец
    if 'num' not in df.columns:
        print("Столбец 'num' (target) не найден — проверь CSV.")
        sys.exit(1)

    # Цель: болезнь сердца
    y = (df['num'] > 0).astype(int)
    X = df.drop(columns=['num'])

    # Кодирование категориальных признаков
    cat_cols = X.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        print("Кодируем категориальные признаки:", list(cat_cols))
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделение выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Обучение логистической регрессии
    clf = LogisticRegression(max_iter=1000, solver='liblinear', random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Метрики
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n=== Метрики ===")
    print(f"Accuracy  = {acc:.4f}")
    print(f"Precision = {prec:.4f}")
    print(f"Recall    = {rec:.4f}")
    print(f"F1-score  = {f1:.4f}")

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha='center', va='center', color='black')
    plt.colorbar(im)
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nМатрица ошибок сохранена в: {OUT_PNG}")

if __name__ == "__main__":
    main()
