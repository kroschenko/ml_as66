import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")  # аккуратный стиль графиков

# ========== ЗАДАНИЕ 1 ==========
print("\n" + "="*30)
print("ЗАДАНИЕ 1: Загрузка данных и первичный анализ")
print("="*30)

df = pd.read_csv("titanic.csv")
print("\nПервые 5 строк:")
print(df.head().to_string(index=False))
print("\nИнформация о данных:")
print(df.info())

# ========== ЗАДАНИЕ 2 ==========
print("\n" + "="*30)
print("ЗАДАНИЕ 2: Визуализация количества выживших и погибших")
print("="*30)

survived_counts = df['Survived'].value_counts()
plt.figure(figsize=(7,5))
bars = plt.bar(['Погибшие', 'Выжившие'], survived_counts, color=['#d9534f', '#5cb85c'])
plt.title("Распределение выживших и погибших пассажиров", fontsize=14, fontweight='bold')
plt.ylabel("Количество пассажиров", fontsize=12)
plt.xlabel("Статус", fontsize=12)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 10, str(yval),
             ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.show()

# ========== ЗАДАНИЕ 3 ==========
print("\n" + "="*30)
print("ЗАДАНИЕ 3: Обработка пропусков в Age (заполнение медианой)")
print("="*30)

print("Количество пропусков в Age до обработки:", df['Age'].isna().sum())
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
print("Количество пропусков в Age после обработки:", df['Age'].isna().sum())
print(f"Медианное значение, использованное для заполнения: {median_age:.1f}")

# ========== ЗАДАНИЕ 4 ==========
print("\n" + "="*30)
print("ЗАДАНИЕ 4: One-Hot Encoding для категориальных признаков")
print("="*30)

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
print("Категориальные признаки преобразованы.")
print("Новые признаки:", [col for col in df.columns if 'Sex_' in col or 'Embarked_' in col])

print("\nПервые строки после кодирования:")
print(df[['Sex_male', 'Embarked_Q', 'Embarked_S']].head(10).to_string(index=False))

# ========== ЗАДАНИЕ 5 ==========
print("\n" + "="*30)
print("ЗАДАНИЕ 5: Гистограмма распределения возрастов")
print("="*30)

plt.figure(figsize=(7,5))
plt.hist(df['Age'], bins=20, color='#5bc0de', edgecolor='black')
plt.title("Распределение возрастов пассажиров", fontsize=14, fontweight='bold')
plt.xlabel("Возраст", fontsize=12)
plt.ylabel("Количество пассажиров", fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ========== ЗАДАНИЕ 6 ==========
print("\n" + "="*30)
print("ЗАДАНИЕ 6: Новый признак FamilySize")
print("="*30)

df['FamilySize'] = df['SibSp'] + df['Parch']
print("Пример новых данных:")
print(df[['SibSp','Parch','FamilySize']].head().to_string(index=False))

# ========== СОХРАНЕНИЕ ==========
print("\n" + "="*30)
print("СОХРАНЕНИЕ ОБРАБОТАННОГО ДАТАСЕТА")
print("="*30)

df.to_csv("titanic_after.csv", index=False)
print("Файл titanic_after.csv сохранён с обработанными данными.")
