import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', None)

df = pd.read_csv("C:\Учёба\ОМО\student-por.csv", sep=";", quotechar='"')

print("Размер данных:", df.shape)

print("Информация о данных:\n", df[["studytime", "failures", "G1", "G2", "G3"]])

X = df[["studytime", "failures", "G1", "G2"]]
y = df["G3"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x="G2", y="G3", data=df, color="blue", alpha=0.6, label="Фактические данные")

reg_line = LinearRegression()
reg_line.fit(df[["G2"]], df["G3"])
x_range = pd.DataFrame({"G2": sorted(df["G2"].unique())})
y_pred_line = reg_line.predict(x_range)
plt.plot(x_range, y_pred_line, color="red", linewidth=2, label="Линия регрессии")

plt.title("Зависимость итоговой оценки (G3) от G2")
plt.xlabel("Оценка за предыдущий период (G2)")
plt.ylabel("Итоговая оценка (G3)")
plt.legend()
plt.grid(alpha=0.3)

plt.show()