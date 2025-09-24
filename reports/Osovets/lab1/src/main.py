import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# -----------------------------
# 1. Определение пути к файлу
# -----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))

# Корень проекта — на пять уровней выше
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..', '..'))

file_path = os.path.join(project_root, 'auto-mpg.csv')
print("Читаем файл из:", file_path)

# Названия колонок
columns = [
    'mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
    'acceleration', 'model_year', 'origin', 'car_name'
]

# Чтение датасета
df = pd.read_csv(
    file_path,
    sep=",",
    names=columns,
    na_values='?',
    header=None
)

# --- Исправляем типы данных ---
numeric_cols = [
    'mpg', 'cylinders', 'displacement', 'horsepower',
    'weight', 'acceleration', 'model_year', 'origin'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Преобразуем model_year в полный год сразу
df['model_year'] = 1900 + df['model_year']

# car_name — строка
df['car_name'] = df['car_name'].astype(str)

# Заполняем пропуски средними значениями
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# --- Новый признак age ---
current_year = datetime.now().year
df['age'] = current_year - df['model_year']

# --- Полный вывод всех колонок ---
pd.set_option('display.max_columns', None)

# --- Выводим информацию ---
print("\nТипы данных:")
print(df.dtypes)

print("\nКоличество пропусков:")
print(df.isnull().sum())

print("\nОсновные статистические показатели:")
print(df.describe())

print("\nПример новых данных (model_year и age):")
print(df[['model_year', 'age']].head())

# --- Графики ---
# 1. Зависимость расхода топлива от веса
fig1, ax1 = plt.subplots(figsize=(8,6))
ax1.scatter(df['weight'], df['mpg'], alpha=0.7)
ax1.set_xlabel("Вес")
ax1.set_ylabel("MPG (миль на галлон)")
ax1.set_title("Зависимость расхода топлива от веса автомобиля")
ax1.grid(True)
plt.show()
plt.close(fig1)

# 2. Гистограмма распределения мощности
fig2, ax2 = plt.subplots(figsize=(8,6))
ax2.hist(df['horsepower'], bins=30, color='skyblue', edgecolor='black')
ax2.set_xlabel("Horsepower")
ax2.set_ylabel("Количество автомобилей")
ax2.set_title("Распределение мощности автомобилей")
ax2.grid(True)
plt.show()
plt.close(fig2)

# 3. Boxplot для сравнения MPG по количеству цилиндров
fig3, ax3 = plt.subplots(figsize=(8,6))
df.boxplot(column='mpg', by='cylinders', ax=ax3)
ax3.set_xlabel("Цилиндры")
ax3.set_ylabel("MPG")
ax3.set_title("MPG по количеству цилиндров")
plt.suptitle("")  # убираем автоматический заголовок
ax3.grid(True)
plt.show()
plt.close(fig3)

# 4. Гистограмма распределения возраста автомобилей
fig4, ax4 = plt.subplots(figsize=(8,6))
ax4.hist(df['age'], bins=20, color='lightgreen', edgecolor='black')
ax4.set_xlabel("Возраст автомобиля (лет)")
ax4.set_ylabel("Количество автомобилей")
ax4.set_title("Распределение возраста автомобилей")
ax4.grid(True)
plt.show()
plt.close(fig4)

# 5. Дополнительно: зависимость MPG от возраста
fig5, ax5 = plt.subplots(figsize=(8,6))
ax5.scatter(df['age'], df['mpg'], alpha=0.7, color='orange')
ax5.set_xlabel("Возраст автомобиля (лет)")
ax5.set_ylabel("MPG (миль на галлон)")
ax5.set_title("Зависимость расхода топлива от возраста автомобиля")
ax5.grid(True)
plt.show()
plt.close(fig5)
