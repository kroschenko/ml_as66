import numpy as np  # Импорт библиотеки NumPy для работы с массивами и числовыми операциями
import pandas as pd  # Импорт библиотеки pandas для загрузки и обработки табличных данных
import matplotlib.pyplot as plt  # Импорт библиотеки matplotlib для построения графиков
# Импорт функций из scikit-learn для разделения данных, кросс-валидации и настройки разбиений
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# Импорт стандартизатора признаков — приводит признаки к одному масштабу
from sklearn.preprocessing import StandardScaler
# Импорт Pipeline — позволяет объединить несколько шагов (например, масштабирование + модель) в одну цепочку
from sklearn.pipeline import Pipeline
# Импорт трёх моделей классификации
from sklearn.neighbors import KNeighborsClassifier  # Метод k-ближайших соседей
from sklearn.tree import DecisionTreeClassifier     # Дерево решений
from sklearn.svm import SVC                         # Метод опорных векторов (Support Vector Machine)
# Импорт метрики accuracy — точность классификации
from sklearn.metrics import accuracy_score

# 1. Загрузка данных
url = 'https://drive.google.com/uc?id=1WNG53OkZav0xxCI_D7Q5qNeERxMdRPfM'
df = pd.read_csv(url)

# Преобразование целевой переменной в числа
df['variety'] = df['variety'].astype('category').cat.codes

# Разделение на признаки и целевую переменную
X = df.drop(columns='variety')
y = df['variety']

# Фиксируем random_state для воспроизводимости
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Подбор оптимального k с помощью кросс-валидации на обучающей выборке
k_values = range(1, 21)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

mean_scores = []
std_scores = []

for k in k_values:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k))
    ])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')
    mean_scores.append(scores.mean())
    std_scores.append(scores.std())

mean_scores = np.array(mean_scores)
std_scores = np.array(std_scores)

best_k = k_values[int(np.argmax(mean_scores))]
print(f"Лучшее значение k по кросс-валидации (на обучающей выборке): {best_k}")

# График: средняя точность ± std
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), mean_scores, marker='o', label='mean CV accuracy')
plt.fill_between(list(k_values), mean_scores - std_scores, mean_scores + std_scores, alpha=0.2)
plt.scatter([best_k], [mean_scores[int(np.argmax(mean_scores))]], color='red', zorder=5, label=f'best k={best_k}')
plt.xticks(list(k_values))
plt.xlabel('k (количество соседей)')
plt.ylabel('CV Accuracy (mean)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 3. Обучение финальных моделей и оценка на тесте
models = {
    'k-NN': Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=best_k))]),
    'Decision Tree': Pipeline([('scaler', StandardScaler()), ('dt', DecisionTreeClassifier(random_state=42))]),
    'SVM': Pipeline([('scaler', StandardScaler()), ('svm', SVC())])
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy on test set: {acc:.4f}")

# 4. Визуализация сравнения
plt.figure(figsize=(7, 4))
names = list(results.keys())
values = [results[n] for n in names]
bars = plt.bar(names, values, color=['tab:blue', 'tab:green', 'tab:orange'])
plt.ylim(0.0, 1.0)
plt.ylabel('Accuracy (test)')
plt.title('Сравнение точности моделей на тестовой выборке')

# Аннотация точности над столбиками
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 5. Вывод лучшей модели по тестовой точности
best_model_name = max(results, key=results.get)
print(f"Лучшая модель по точности на тесте: {best_model_name} (accuracy = {results[best_model_name]:.4f})")