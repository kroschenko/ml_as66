import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

data = pd.read_csv("Fish.csv")

X = data[["Length1", "Length2", "Length3", "Height", "Width"]]
y = data["Weight"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred)

print("Качество модели:")
print("R² =", round(r2, 2))
print("RMSE =", round(rmse, 2))

plt.scatter(data["Length3"], data["Weight"], alpha=0.6, label="Данные")
plt.plot(data["Length3"], model.predict(data[["Length1","Length2","Length3","Height","Width"]]),
         color="red", label="Линия регрессии")
plt.xlabel("Length3")
plt.ylabel("Weight")
plt.title("Зависимость длины (Length3) и веса (Weight)")
plt.legend()
plt.show()
