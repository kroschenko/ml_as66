import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('breast_cancer.csv')

print("Пропущенные значения в данных:")
print(df.isnull().sum())

df = df.drop('Unnamed: 32', axis=1)

print(f"\nРазмер данных после удаления пустого столбца: {df.shape}")

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']

print(f"\nРазмер признаков: {X.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Обучающая выборка: {X_train.shape}")
print(f"Тестовая выборка: {X_test.shape}")

models = {
    'k-NN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

results = {}
predictions = {}

for name, model in models.items():
    print(f"Обучение {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Доброкач. (0)', 'Злокач. (1)'],
                yticklabels=['Доброкач. (0)', 'Злокач. (1)'])
    plt.title(f'Матрица ошибок - {model_name}')
    plt.ylabel('Истинные значения')
    plt.xlabel('Предсказанные значения')
    plt.show()


for name in models.keys():
    plot_confusion_matrix(y_test, predictions[name], name)
    print(f"{name} - Отчет по классификации:")
    print(classification_report(y_test, predictions[name],
                                target_names=['Доброкачественная', 'Злокачественная']))

comparison_df = pd.DataFrame(results).T
print("Сравнение метрик для класса 'Злокачественная':")
print(comparison_df)

metrics = ['precision', 'recall', 'f1_score']
x = np.arange(len(metrics))
width = 0.25

plt.figure(figsize=(12, 6))
for i, (model, color) in enumerate(zip(models.keys(), ['blue', 'orange', 'green'])):
    values = [results[model][metric] for metric in metrics]
    plt.bar(x + i * width, values, width, label=model, color=color, alpha=0.7)

plt.xlabel('Метрики')
plt.ylabel('Значение')
plt.title('Сравнение моделей для класса "Злокачественная"')
plt.xticks(x + width, ['Точность', 'Полнота', 'F1-мера'])
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

best_recall_model = max(results.items(), key=lambda x: x[1]['recall'])
best_f1_model = max(results.items(), key=lambda x: x[1]['f1_score'])

print(f"Лучшая модель для минимизации ложноотрицательных: {best_recall_model[0]}")
print(f"Полнота: {best_recall_model[1]['recall']:.4f}")

print("\nАнализ ложноотрицательных прогнозов:")
for name in models.keys():
    cm = confusion_matrix(y_test, predictions[name])
    fn = cm[1, 0]
    total_malignant = cm[1, 0] + cm[1, 1]
    fn_rate = fn / total_malignant
    print(f"{name}: {fn} ложноотрицательных ({fn_rate:.2%})")

print(f"\nРекомендуемая модель: {best_recall_model[0]}")
print(f"Полнота: {best_recall_model[1]['recall']:.4f}")