import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

df_heart = pd.read_csv('heart_disease_uci.csv')

df_heart['heart_disease'] = (df_heart['num'] > 0).astype(int)

numeric_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

for col in numeric_features:
    if col in df_heart.columns:
        df_heart[col].fillna(df_heart[col].median(), inplace=True)

for col in categorical_features:
    if col in df_heart.columns:
        df_heart[col].fillna(df_heart[col].mode()[0] if not df_heart[col].mode().empty else 'unknown', inplace=True)

for col in categorical_features:
    if col in df_heart.columns:
        le = LabelEncoder()
        df_heart[col + '_encoded'] = le.fit_transform(df_heart[col].astype(str))

feature_columns = numeric_features + [col + '_encoded' for col in categorical_features]
feature_columns = [col for col in feature_columns if col in df_heart.columns]

X_clf = df_heart[feature_columns]
y_clf = df_heart['heart_disease']

scaler = StandardScaler()
X_clf_scaled = scaler.fit_transform(X_clf)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf_scaled, y_clf, test_size=0.3, random_state=42, stratify=y_clf
)

logreg_model = LogisticRegression(random_state=42, max_iter=1000)
logreg_model.fit(X_train_clf, y_train_clf)

y_pred_clf = logreg_model.predict(X_test_clf)

accuracy = accuracy_score(y_test_clf, y_pred_clf)
precision = precision_score(y_test_clf, y_pred_clf)
recall = recall_score(y_test_clf, y_pred_clf)
f1 = f1_score(y_test_clf, y_pred_clf)

print("КЛАССИФИКАЦИЯ  БОЛЕЗНИ СЕРДЦА - classif.py:53")
print(f"Accuracy: {accuracy:.4f} - classif.py:54")
print(f"Precision: {precision:.4f} - classif.py:55")
print(f"Recall: {recall:.4f} - classif.py:56")
print(f"F1Score: {f1:.4f} - classif.py:57")

cm = confusion_matrix(y_test_clf, y_pred_clf)
print("\nМатрица ошибок: - classif.py:60")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Нет болезни', 'Есть болезнь'],
           yticklabels=['Нет болезни', 'Есть болезнь'])
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок - Диагностика заболеваний сердца')
plt.savefig('heart_disease_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
plt.ylabel('Значение метрики')
plt.title('Метрики классификации заболеваний сердца')
plt.ylim(0, 1)
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
plt.savefig('heart_disease_metrics.png', dpi=300, bbox_inches='tight')
plt.show()