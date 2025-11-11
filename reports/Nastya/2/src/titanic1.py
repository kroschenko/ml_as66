import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("california_housing.csv")
df.dropna(inplace=True)
df["ocean_proximity"] = df["ocean_proximity"].astype("category").cat.codes


y = df["median_house_value"]


X_single = df[["median_income"]]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_single, y, test_size=0.2, random_state=42)
model_single = LinearRegression()
model_single.fit(X_train_s, y_train_s)
y_pred_s = model_single.predict(X_test_s)
mse_s = mean_squared_error(y_test_s, y_pred_s)
r2_s = r2_score(y_test_s, y_pred_s)


X_multi = df.drop(columns=["median_house_value"])
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)
model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)
y_pred_m = model_multi.predict(X_test_m)
mse_m = mean_squared_error(y_test_m, y_pred_m)
r2_m = r2_score(y_test_m, y_pred_m)


print("Однопараметрическая модель:")
print(f"  MSE: {mse_s:.2f}")
print(f"  R²: {r2_s:.2f}")
print("\nМногопараметрическая модель:")
print(f"  MSE: {mse_m:.2f}")
print(f"  R²: {r2_m:.2f}")


plt.figure(figsize=(10, 6))
plt.scatter(X_test_s, y_test_s, alpha=0.3, label="Фактические значения")
plt.plot(X_test_s, y_pred_s, color="red", label="Линия регрессии")
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Однопараметрическая регрессия: Доход vs Стоимость жилья")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
