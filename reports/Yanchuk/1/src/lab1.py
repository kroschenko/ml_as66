import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

iris = sns.load_dataset("iris")

iris.rename(columns={
    "sepal_length":  "sepal.length",
    "sepal_width":   "sepal.width",
    "petal_length":  "petal.length",
    "petal_width":   "petal.width",
    "species":       "variety"
}, inplace=True)

print("Shape:", iris.shape)
print("Columns:", iris.columns)
print(iris.head())

missing = iris.isnull().sum()
counts  = iris["variety"].value_counts()
means   = iris.groupby("variety").mean(numeric_only=True)

features = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
iris_scaled_df = iris.copy()
iris_scaled_df[features] = (iris[features] - iris[features].mean()) / iris[features].std()

iris_encoded = pd.get_dummies(iris_scaled_df, columns=["variety"])

report_path = Path(__file__).parent / "iris_report.txt"
with report_path.open("w", encoding="utf-8") as f:
    f.write("Исходные данные (все строки):\n")
    f.write(iris.to_string(index=False) + "\n\n")
    f.write("Проверка пропущенных значений:\n")
    f.write(missing.to_string() + "\n\n")
    f.write("Количество образцов по каждому виду:\n")
    f.write(counts.to_string() + "\n\n")
    f.write("Средние значения признаков по каждому виду:\n")
    f.write(means.to_string() + "\n\n")
    f.write("Стандартизованные данные (первые 5 строк):\n")
    f.write(iris_scaled_df.head().to_string(index=False) + "\n\n")
    f.write("One-Hot Encoding (первые 5 строк):\n")
    f.write(iris_encoded.head().to_string(index=False) + "\n\n")
print(f"Отчёт сохранён в {report_path.name}")

sns.pairplot(iris, hue="variety", diag_kind="kde")
plt.suptitle("Pair Plot признаков Iris", y=1.02)
plt.savefig("pairplot.png")
plt.close()

plt.figure(figsize=(8, 6))
sns.boxplot(x="variety", y="petal.length", data=iris)
plt.title("Box Plot: Petal Length по видам ириса")
plt.savefig("boxplot.png")
plt.close()

print("Графики сохранены в pairplot.png и boxplot.png")
