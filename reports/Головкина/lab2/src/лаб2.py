import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

# § Загрузка данных
local_file = Path(r"D:\у--ба\3 kurs\омо\2\california_housing.csv")
if not local_file.exists():
    raise FileNotFoundError(f"Файл не найден: {local_file}")

df = pd.read_csv(local_file)
print(f"Данные успешно загружены из: {local_file}")

# § Предобработка
target_column = "median_house_value"
if target_column not in df.columns:
    raise ValueError(f"Столбец '{target_column}' не найден в датасете.")

# Оставляем только числовые признаки
df_numeric = df.select_dtypes(include=["number"])

# Заполняем пропуски медианой
df_filled = df_numeric.fillna(df_numeric.median(numeric_only=True))

# § Разделение на признаки и целевую переменную
X = df_filled.drop(columns=target_column)
y = df_filled[target_column]

# § Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# § Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# § Предсказания
y_pred = model.predict(X_test)

# § Оценка качества модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMSE: {mse:.2f}")
print(f"R²: {r2:.2f}")

# § Визуализация зависимости от median_income
if "median_income" in X.columns:
    # Обучаем отдельную модель только по median_income
    X_income = df_filled[["median_income"]]
    y_value = df_filled[target_column]

    X_income_train, X_income_test, y_value_train, y_value_test = train_test_split(
        X_income, y_value, test_size=0.2, random_state=42
    )

    income_model = LinearRegression()
    income_model.fit(X_income_train, y_value_train)

    # Создаём точки для линии регрессии
    income_range = pd.DataFrame({
        "median_income": sorted(X_income_test["median_income"].unique())
    })
    predicted_values = income_model.predict(income_range)

    # Строим график
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_income_test["median_income"], y=y_value_test, label="Фактические значения")
    plt.plot(income_range["median_income"], predicted_values, color="red", label="Линия регрессии")

    plt.xlabel("Median Income")
    plt.ylabel("Median House Value")
    plt.title("Зависимость стоимости жилья от дохода")
    plt.legend()
    plt.tight_layout()
    plt.show()
