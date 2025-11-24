import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("student-por.csv", sep=';')

print(f"Размерность данных: {df.shape}")
print("\nПервые 5 строк данных:")
print(df.head())

features = ['studytime', 'failures', 'G1', 'G2']
target = 'G3'

X = df[features]
y = df[target]

print(f"\nПризнаки (X): {X.shape}")
print(f"Целевая переменная (y): {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nРазмер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nРЕЗУЛЬТАТЫ МОДЕЛИ")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}")

print(f"\nКоэффициенты модели:")
for i, feature in enumerate(features):
    print(f"  {feature}: {model.coef_[i]:.4f}")
print(f"Свободный член (intercept): {model.intercept_:.4f}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_test['G2'], y_test, alpha=0.7, color='blue', label='Фактические значения')
plt.scatter(X_test['G2'], y_pred, alpha=0.7, color='red', label='Предсказанные значения')

sorted_indices = np.argsort(X_test['G2'].values)
G2_sorted = X_test['G2'].iloc[sorted_indices]
pred_sorted = y_pred[sorted_indices]

plt.plot(G2_sorted, pred_sorted, color='black', linewidth=2, label='Линия регрессии')
plt.xlabel('G2 (Оценка за второй период)')
plt.ylabel('G3 (Итоговая оценка)')
plt.title('Зависимость G3 от G2 с линией регрессии')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Фактические значения G3')
plt.ylabel('Предсказанные значения G3')
plt.title('Фактические vs Предсказанные значения')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Предсказанные значения G3')
plt.ylabel('Остатки')
plt.title('Остатки модели')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ")
print(f"Среднее значение G3 в тестовой выборке: {y_test.mean():.2f}")
print(f"Стандартное отклонение G3 в тестовой выборке: {y_test.std():.2f}")
print(f"MAE относительно среднего: {(mae / y_test.mean() * 100):.2f}%")

print(f"\nПРИМЕР ПРЕДСКАЗАНИЯ")
sample_data = np.array([[2, 0, 14, 15]])
prediction = model.predict(sample_data)
print(f"Для студента с параметрами: studytime=2, failures=0, G1=14, G2=15")
print(f"Предсказанная итоговая оценка (G3): {prediction[0]:.2f}")

feature_importance = pd.DataFrame({
    'Признак': features,
    'Коэффициент': model.coef_,
    'Абсолютное значение': np.abs(model.coef_)
}).sort_values('Абсолютное значение', ascending=False)

print(f"\nВАЖНОСТЬ ПРИЗНАКОВ")
print(feature_importance)

print(f"\nКОРРЕЛЯЦИОННАЯ МАТРИЦА")
correlation_matrix = df[features + [target]].corr()
print(correlation_matrix)