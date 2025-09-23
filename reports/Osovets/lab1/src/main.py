import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Загрузка данных
# -----------------------------
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
           'acceleration', 'model_year', 'origin', 'car_name']

df = pd.read_csv(url, names=columns, sep=r'\s+', na_values='?')

# -----------------------------
# 2. Обработка пропусков
# -----------------------------
df['horsepower'] = df['horsepower'].astype(float)
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].mean())

# -----------------------------
# 3. Исследовательский анализ
# -----------------------------
print("Типы данных:\n", df.dtypes)
print("\nКоличество пропусков:\n", df.isna().sum())
print("\nОсновные статистические показатели:\n", df.describe())

# -----------------------------
# 4. Преобразование категориального признака
# -----------------------------
df = pd.get_dummies(df, columns=['origin'], prefix='origin')

# -----------------------------
# 5. Создание нового признака age
# -----------------------------
df['age'] = 1983 - (1900 + df['model_year'])

# -----------------------------
# 6. Визуализация данных
# -----------------------------

# 6.1 Расход топлива vs вес
plt.figure(figsize=(8,6))
plt.scatter(df['weight'], df['mpg'], alpha=0.7)
plt.title('mpg vs weight')
plt.xlabel('Вес автомобиля')
plt.ylabel('Расход топлива (mpg)')
plt.grid(True)
plt.show()

# 6.2 Распределение количества цилиндров
cylinder_counts = df['cylinders'].value_counts().sort_index()
plt.figure(figsize=(8,6))
plt.bar(cylinder_counts.index, cylinder_counts.values, color='skyblue')
plt.title('Распределение цилиндров')
plt.xlabel('Количество цилиндров')
plt.ylabel('Количество автомобилей')
plt.show()

# 6.3 Гистограмма расхода топлива
plt.figure(figsize=(8,6))
plt.hist(df['mpg'], bins=15, color='lightgreen', edgecolor='black')
plt.title('Распределение расхода топлива (mpg)')
plt.xlabel('mpg')
plt.ylabel('Количество автомобилей')
plt.show()

# 6.4 Расход топлива vs horsepower
plt.figure(figsize=(8,6))
plt.scatter(df['horsepower'], df['mpg'], color='orange', alpha=0.7)
plt.title('mpg vs horsepower')
plt.xlabel('Мощность (horsepower)')
plt.ylabel('Расход топлива (mpg)')
plt.grid(True)
plt.show()

# 6.5 Расход топлива vs displacement
plt.figure(figsize=(8,6))
plt.scatter(df['displacement'], df['mpg'], color='red', alpha=0.7)
plt.title('mpg vs displacement')
plt.xlabel('Объём двигателя (displacement)')
plt.ylabel('Расход топлива (mpg)')
plt.grid(True)
plt.show()

# 6.6 Корреляционная матрица
plt.figure(figsize=(10,8))
corr = df[['mpg','cylinders','displacement','horsepower','weight','acceleration','age']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Корреляционная матрица')
plt.show()

# 6.7 Boxplot веса автомобиля по количеству цилиндров
plt.figure(figsize=(8,6))
sns.boxplot(x='cylinders', y='weight', data=df, hue='cylinders', legend=False, palette='Set2')
plt.title('Вес автомобиля по количеству цилиндров')
plt.xlabel('Количество цилиндров')
plt.ylabel('Вес автомобиля')
plt.show()
