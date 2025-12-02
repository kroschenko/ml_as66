import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Загрузка данных и вывод информации
data = pd.read_csv('heart.csv')
print(data.info())
print(data.isnull().sum())

# 2. Столбчатая диаграмма количества здоровых и больных пациентов
plt.figure(figsize=(8, 5))
sns.countplot(x='target', data=data)
plt.title('Количество здоровых и больных пациентов')
plt.xlabel('Наличие болезни (0 = здоров, 1 = больной)')
plt.ylabel('Количество')
plt.xticks([0, 1], ['Здоровые', 'Больные'])
plt.show()

# 3. Диаграмма рассеяния для максимального пульса от возраста
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='thalach', hue='target', data=data, palette={0: 'blue', 1: 'red'})
plt.title('Зависимость максимального пульса от возраста')
plt.xlabel('Возраст')
plt.ylabel('Максимальный пульс')
plt.legend(title='Наличие болезни', labels=['Здоровые', 'Больные'])
plt.show()

# 4. Преобразование признака 'sex' и One-Hot Encoding
data['sex'] = data['sex'].replace({0: 'female', 1: 'male'})
encoder = OneHotEncoder(sparse=False)
sex_encoded = encoder.fit_transform(data[['sex']])
sex_df = pd.DataFrame(sex_encoded, columns=['female', 'male'])
data = pd.concat([data, sex_df], axis=1)

# 5. Средний уровень холестерина для больных и здоровых пациентов
mean_chol = data.groupby('target')['chol'].mean()
print(mean_chol)

# 6. Нормализация признаков
scaler = StandardScaler()
data[['age', 'trestbps', 'chol', 'thalach']] = scaler.fit_transform(data[['age', 'trestbps', 'chol', 'thalach']])
print(data[['age', 'trestbps', 'chol', 'thalach']].head())