import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("G:/ЛАБЫ 3 КУРС/ОМО/titanic.csv")

print("🔍 Первые 5 строк выборки:")
print(df.head())

print("\n📋 Информация о столбцах:")
df.info()

survival_counts = df["Survived"].value_counts() 

plt.figure(figsize=(6, 4))
sns.barplot(x=survival_counts.index, y=survival_counts.values, palette="viridis")
plt.xticks([0, 1], ["Погибли", "Выжили"])
plt.title("Количество выживших и погибших пассажиров")
plt.xlabel("Статус")
plt.ylabel("Количество")
plt.savefig("G:/ЛАБЫ 3 КУРС/ОМО/survival_barplot.png")
plt.close()

age_missing_before = df["Age"].isnull().sum()
df["Age"].fillna(df["Age"].median(), inplace=True)
age_missing_after = df["Age"].isnull().sum()

df_encoded = pd.get_dummies(df, columns=["Sex", "Embarked"])

plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=30, kde=True, color="steelblue")
plt.title("Распределение возрастов пассажиров")
plt.xlabel("Возраст")
plt.ylabel("Количество")
plt.savefig("G:/ЛАБЫ 3 КУРС/ОМО/age_histogram.png")
plt.close()

df["FamilySize"] = df["SibSp"] + df["Parch"]

with open("G:/ЛАБЫ 3 КУРС/ОМО/titanic_report.txt", "w", encoding="utf-8") as f:
    f.write("🔍 Первые 5 строк:\n")
    f.write(df.head().to_string() + "\n\n")
    f.write("📋 Информация о столбцах:\n")
    f.write(str(df.info()) + "\n\n")
    f.write("📊 Количество выживших и погибших:\n")
    f.write(str(survival_counts) + "\n\n")
    f.write(f"🔧 Пропущенные значения в Age до: {age_missing_before}, после: {age_missing_after}\n\n")
    f.write("📐 Данные после One-Hot Encoding (первые 5 строк):\n")
    f.write(df_encoded.head().to_string() + "\n\n")
    f.write("👨‍👩‍👧 Новый признак FamilySize (первые 5 строк):\n")
    f.write(df[["SibSp", "Parch", "FamilySize"]].head().to_string() + "\n\n")

print("✅ Отчёт сохранён в titanic_report.txt")
print("📊 Графики сохранены в survival_barplot.png и age_histogram.png")
