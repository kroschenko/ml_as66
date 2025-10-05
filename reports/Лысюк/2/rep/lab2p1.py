import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка данных
df = pd.read_csv('fish.csv')
print(df.head())

# Объявление признаков и целевой переменной
features = ['Length1', 'Length2', 'Length3', 'Height', 'Width']
X = df[features]
y = df['Weight']

# Обучение модели линейной регрессии на всех признаках
model = LinearRegression()
model.fit(X, y)

# Предсказания и метрики
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f'R2: {r2:.4f}')
print(f'RMSE: {rmse:.4f}')

# Извлечение коэффициентов и intercept из модели
coef = model.coef_
intercept = model.intercept_

# Коэффициент для признака Length3
coef_length3 = coef[features.index('Length3')]

# Создаем значения x для построения линии (границы диапазона Length3)
x_vals = np.array([df['Length3'].min(), df['Length3'].max()])

# Вычисляем предсказания y на основе только Length3
y_vals = intercept + coef_length3 * x_vals

# Визуализация
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Length3'], y=y, label='Actual')
plt.plot(x_vals, y_vals, color='red', label='Regression line (из модели)')
plt.xlabel('Length3')
plt.ylabel('Weight')
plt.title('Weight vs Length3 with Regression Line (из модели)')
plt.legend()
plt.show()
