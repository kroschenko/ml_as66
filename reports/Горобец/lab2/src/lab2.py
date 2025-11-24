import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Загрузка данных
df = pd.read_csv('C:/Users/Anton/Downloads/auto-mpg.csv')  # Укажи путь к файлу, если он не в текущей папке

# 2. Обработка пропусков
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')  # '?' → NaN
df.dropna(subset=['cylinders', 'horsepower', 'weight', 'mpg'], inplace=True)

# 3. Обработка категориальных признаков
df['origin'] = df['origin'].astype('category')
df = pd.get_dummies(df, columns=['origin'], drop_first=True)

# 4. Выбор признаков и целевой переменной
X = df[['cylinders', 'horsepower', 'weight']]
y = df['mpg']

# 5. Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Предсказание и метрики
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse:.2f}')
print(f'R²: {r2:.2f}')

# 8. Визуализация зависимости mpg от horsepower
plt.figure(figsize=(10, 6))
plt.scatter(df['horsepower'], df['mpg'], alpha=0.5, label='Фактические данные')

# Линия регрессии при фиксированных средних значениях других признаков
hp_range = np.linspace(df['horsepower'].min(), df['horsepower'].max(), 100)
mean_cyl = df['cylinders'].mean()
mean_weight = df['weight'].mean()
X_line = pd.DataFrame({'cylinders': mean_cyl, 'horsepower': hp_range, 'weight': mean_weight})
y_line = model.predict(X_line)

plt.plot(hp_range, y_line, color='red', label='Линия регрессии')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title('Зависимость MPG от Horsepower')
plt.legend()
plt.grid(True)
plt.show()
