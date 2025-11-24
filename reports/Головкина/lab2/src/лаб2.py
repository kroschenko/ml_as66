import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

local_file = Path(r"D:\у--ба\3 kurs\омо\2\california_housing.csv")

if not local_file.exists():
    raise FileNotFoundError(f" Файл не найден: {local_file}")

try:
    df = pd.read_csv(local_file)
    print(f" Данные успешно загружены из локального файла: {local_file}")
except Exception as e:
    raise RuntimeError(f"Ошибка при чтении файла: {e}")

target_column = "median_house_value"
if target_column not in df.columns:
    raise ValueError(f" Столбец '{target_column}' не найден в датасете.")

# Уд нечисловых призн 
df_numeric = df.select_dtypes(include=["number"])

df_filled = df_numeric.fillna(df_numeric.median(numeric_only=True))

X = df_filled.drop(columns=target_column)
y = df_filled[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n MSE: {mse:.2f}")
print(f" R²: {r2:.2f}")

if "median_income" in X_test.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_test["median_income"], y=y_test, label="Фактические значения")

    plot_data = pd.DataFrame({
        "median_income": X_test["median_income"],
        "Predicted": y_pred
    }).sort_values("median_income")

    sns.lineplot(x=plot_data["median_income"], y=plot_data["Predicted"], color="red", label="Линия регрессии")

    plt.xlabel("median_income")
    plt.ylabel("Median House Value")
    plt.title("Зависимость стоимости жилья от дохода")
    plt.legend()
    plt.tight_layout()
    plt.show()