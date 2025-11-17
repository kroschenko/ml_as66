import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Настройка отображения графиков
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 6)

# 1. ЗАГРУЗКА ДАННЫХ
print("=" * 50)
print("1. ЗАГРУЗКА ДАННЫХ")
print("=" * 50)

df = pd.read_csv('../../../../../pima-indians-diabetes.csv', sep=',', comment='#')

column_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
df.columns = column_names

print("Первые 5 строк данных:")
print(df.head())
print("\nИнформация о данных:")
print(df.info())

# 2. СТАТИСТИЧЕСКИЙ АНАЛИЗ
print("\n" + "=" * 50)
print("2. СТАТИСТИЧЕСКИЕ ХАРАКТЕРИСТИКИ")
print("=" * 50)

print("Основные статистические характеристики:")
print(df.describe())

# Проверка на пропущенные значения (в данном наборе пропуски обозначены как 0 в некоторых столбцах)
print("\nКоличество нулевых значений в каждом столбце:")
for column in df.columns:
    zero_count = (df[column] == 0).sum()
    print(f"{column}: {zero_count} нулевых значений")

# 3. ОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ
print("\n" + "=" * 50)
print("3. ОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ")
print("=" * 50)

# Столбцы, где 0 является некорректным значением
columns_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Замена нулевых значений на медианные (исключая нули при расчете медианы)
for column in columns_to_fix:
    median_value = df[df[column] != 0][column].median()
    df[column] = df[column].replace(0, median_value)
    print(f"Столбец {column}: заменено {(df[column] == 0).sum()} значений на медиану {median_value:.2f}")

print("\nСтатистика после обработки пропусков:")
print(df[columns_to_fix].describe())

# 4. ВИЗУАЛИЗАЦИЯ ДАННЫХ
print("\n" + "=" * 50)
print("4. ВИЗУАЛИЗАЦИЯ ДАННЫХ")
print("=" * 50)

# Создание подграфиков
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 4.1 Гистограмма BMI
axes[0, 0].hist(df['BMI'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Распределение индекса массы тела (BMI)')
axes[0, 0].set_xlabel('BMI')
axes[0, 0].set_ylabel('Частота')

# 4.2 Гистограмма Age
axes[0, 1].hist(df['Age'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Распределение возраста (Age)')
axes[0, 1].set_xlabel('Возраст (лет)')
axes[0, 1].set_ylabel('Частота')

# 4.3 Круговая диаграмма Outcome
outcome_counts = df['Outcome'].value_counts()
labels = ['Нет диабета', 'Есть диабет']
colors = ['lightblue', 'lightcoral']
axes[1, 0].pie(outcome_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
axes[1, 0].set_title('Распределение наличия диабета')

# 4.4 Матрица корреляции
correlation_columns = ['Glucose', 'BMI', 'Age', 'Outcome']
correlation_matrix = df[correlation_columns].corr()
im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
axes[1, 1].set_xticks(range(len(correlation_columns)))
axes[1, 1].set_yticks(range(len(correlation_columns)))
axes[1, 1].set_xticklabels(correlation_columns)
axes[1, 1].set_yticklabels(correlation_columns)

# Добавление значений корреляции на тепловую карту
for i in range(len(correlation_columns)):
    for j in range(len(correlation_columns)):
        text = axes[1, 1].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black")

axes[1, 1].set_title('Матрица корреляции')

plt.tight_layout()
plt.show()

# 5. ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ
print("\n" + "=" * 50)
print("5. ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ")
print("=" * 50)

# Диаграмма рассеяния: Glucose vs BMI с цветом по Outcome
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['Glucose'], df['BMI'], c=df['Outcome'], 
                     cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Outcome (0=Нет, 1=Да)')
plt.xlabel('Уровень глюкозы (Glucose)')
plt.ylabel('Индекс массы тела (BMI)')
plt.title('Зависимость BMI от уровня глюкозы')
plt.grid(True, alpha=0.3)
plt.show()

# 6. СТАНДАРТИЗАЦИЯ ДАННЫХ
print("\n" + "=" * 50)
print("6. СТАНДАРТИЗАЦИЯ ДАННЫХ")
print("=" * 50)

# Признаки для стандартизации (все кроме Outcome)
features_to_standardize = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

df_original = df.copy()

# Стандартизация
scaler = StandardScaler()
df_standardized = df.copy()
df_standardized[features_to_standardize] = scaler.fit_transform(df[features_to_standardize])

print("Данные после стандартизации (первые 5 строк):")
print(df_standardized[features_to_standardize].head())

# Визуализация распределения до и после стандартизации
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# До стандартизации
axes[0].hist(df_original['Glucose'], bins=20, color='skyblue', edgecolor='black', alpha=0.7, label='Исходные')
axes[0].set_title('Распределение Glucose до стандартизации')
axes[0].set_xlabel('Glucose')
axes[0].set_ylabel('Частота')

# После стандартизации
axes[1].hist(df_standardized['Glucose'], bins=20, color='lightcoral', edgecolor='black', alpha=0.7, label='Стандартизированные')
axes[1].set_title('Распределение Glucose после стандартизации')
axes[1].set_xlabel('Glucose (стандартизированный)')
axes[1].set_ylabel('Частота')

plt.tight_layout()
plt.show()

# 7. ВЫВОДЫ И АНАЛИЗ
print("\n" + "=" * 50)
print("7. ВЫВОДЫ И АНАЛИЗ")
print("=" * 50)

print("Ключевые наблюдения:")
print("1. Размер набора данных:", df.shape)
print("2. Распределение классов:")
print(f"   - Без диабета: {outcome_counts[0]} случаев ({outcome_counts[0]/len(df)*100:.1f}%)")
print(f"   - С диабетом: {outcome_counts[1]} случаев ({outcome_counts[1]/len(df)*100:.1f}%)")

# Анализ корреляции
correlation_with_outcome = df[correlation_columns].corr()['Outcome'].sort_values(ascending=False)
print("\n3. Корреляция признаков с Outcome:")
for feature, corr in correlation_with_outcome.items():
    if feature != 'Outcome':
        print(f"   - {feature}: {corr:.3f}")

print("\n4. Статистика по возрасту:")
print(f"   - Средний возраст: {df['Age'].mean():.1f} лет")
print(f"   - Медианный возраст: {df['Age'].median():.1f} лет")
print(f"   - Минимальный возраст: {df['Age'].min()} лет")
print(f"   - Максимальный возраст: {df['Age'].max()} лет")

print("\n5. Статистика по BMI:")
print(f"   - Средний BMI: {df['BMI'].mean():.1f}")
print(f"   - Медианный BMI: {df['BMI'].median():.1f}")

# Сохранение обработанных данных
df_standardized.to_csv('pima_indians_diabetes_processed.csv', index=False)
print("\nОбработанные данные сохранены в файл 'pima_indians_diabetes_processed.csv'")
