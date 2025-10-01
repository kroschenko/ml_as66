# Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io  # для буфера info()

# 1. Загрузка данных
df = pd.read_csv(r"D:\ЛАБЫ\ОМО\Titanic-Dataset.csv")  # используем сырую строку для пути

# --- Задача 1: первые строки и общая информация ---
report1 = "Первые 5 строк датасета:\n" + df.head().to_string() + "\n\n"
report1 += "Общая информация о данных:\n"

buffer = io.StringIO()  # создаем буфер
df.info(buf=buffer)
report1 += buffer.getvalue()  # записываем содержимое info()

with open("report_task1.txt", "w", encoding="utf-8") as f:
    f.write(report1)

# --- Задача 2: количество выживших и погибших ---
survived_counts = df['Survived'].value_counts()

plt.figure(figsize=(6,4))
survived_counts.plot(kind='bar', color=['red','green'])
plt.title("Количество выживших (1) и погибших (0)")
plt.xlabel("Статус выживания")
plt.ylabel("Количество пассажиров")
plt.xticks(rotation=0)
plt.savefig("survival_counts.png")  # сохраняем график
plt.close()

report2 = f"Количество выживших и погибших:\n{survived_counts.to_string()}\nГрафик сохранён в 'survival_counts.png'."
with open("report_task2.txt", "w", encoding="utf-8") as f:
    f.write(report2)

# --- Задача 3: обработка пропусков в Age ---
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)

report3 = f"Медианное значение возраста: {median_age}\nПропуски в столбце 'Age' заполнены медианой."
with open("report_task3.txt", "w", encoding="utf-8") as f:
    f.write(report3)

# --- Задача 4: One-Hot Encoding ---
df = pd.get_dummies(df, columns=['Sex','Embarked'], drop_first=True)

report4 = "Преобразованы категориальные признаки 'Sex' и 'Embarked' в числовые с помощью One-Hot Encoding.\n"
report4 += "Первые строки после преобразования:\n" + df.head().to_string()
with open("report_task4.txt", "w", encoding="utf-8") as f:
    f.write(report4)

# --- Задача 5: гистограмма возрастов ---
plt.figure(figsize=(8,5))
plt.hist(df['Age'], bins=30, color='skyblue', edgecolor='black')
plt.title("Распределение возрастов пассажиров")
plt.xlabel("Возраст")
plt.ylabel("Количество")
plt.savefig("age_distribution.png")  # сохраняем график
plt.close()

report5 = "Построена гистограмма распределения возрастов пассажиров.\nГрафик сохранён в 'age_distribution.png'."
with open("report_task5.txt", "w", encoding="utf-8") as f:
    f.write(report5)

# --- Задача 6: новый признак FamilySize ---
df['FamilySize'] = df['SibSp'] + df['Parch']

report6 = "Создан новый признак 'FamilySize' = SibSp + Parch.\n"
report6 += "Первые строки с новым признаком:\n" + df[['SibSp','Parch','FamilySize']].head().to_string()
with open("report_task6.txt", "w", encoding="utf-8") as f:
    f.write(report6)
