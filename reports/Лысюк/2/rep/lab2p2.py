import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Загрузка и подготовка данных
df = pd.read_csv('bank.csv', sep=';')

categorical_cols = df.select_dtypes(include='object').columns.tolist()
target = 'y'
categorical_cols.remove(target)
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_encoded.drop(target, axis=1)
y = df_encoded[target].map({'no': 0, 'yes': 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
labels = ['no', 'yes']  # метки для классов

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=labels, yticklabels=labels, ax=ax)

ax.set_xlabel('')  # убираем стандартные подписи осей
ax.set_ylabel('')

# Убираем подписи по осям — будем добавлять свои
ax.set_xticklabels([])
ax.set_yticklabels([])

# Подписи сверху (предсказанный класс)
ax.text(0.4, 1.15, 'Предсказанный класс: no', ha='center', va='bottom',
        transform=ax.transAxes, fontsize=10, fontweight='bold')
ax.text(0.75, 1.15, 'Предсказанный класс: yes', ha='center', va='bottom',
        transform=ax.transAxes, fontsize=10, fontweight='bold')

# Подписи снизу (истинный класс)
ax.text(0.4, -0.25, 'Истинный класс: no', ha='center', va='top',
        transform=ax.transAxes, fontsize=10, fontweight='bold')
ax.text(0.75, -0.25, 'Истинный класс: yes', ha='center', va='top',
        transform=ax.transAxes, fontsize=10, fontweight='bold')

ax.set_title('Матрица ошибок')

plt.tight_layout()
plt.show()
