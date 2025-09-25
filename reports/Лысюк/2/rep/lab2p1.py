import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Загрузим файл из текущей папки
df = pd.read_csv('fish.csv')

# Остальной код остается без изменений
print(df.head())

# Объявляем признаки и целевую
features = ['Length1', 'Length2', 'Length3', 'Height', 'Width']
X = df[features]
y = df['Weight']

# Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X, y)

# Предсказания и метрики
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f'R2: {r2:.4f}')
print(f'RMSE: {rmse:.4f}')

# Построение графика Length3 vs Weight с линией регрессии
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Length3'], y=y, label='Actual')
# Обучение модели только по Length3 для линии регрессии
single_feature = df[['Length3']]
model_single = LinearRegression().fit(single_feature, y)
y_pred_single = model_single.predict(single_feature)
sns.lineplot(x=df['Length3'], y=y_pred_single,
             color='red', label='Regression line')
plt.xlabel('Length3')
plt.ylabel('Weight')
plt.title('Weight vs Length3 with Regression Line')
plt.legend()
plt.show()
