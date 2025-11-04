import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("medical_cost_personal_dataset.csv")
df.columns = [c.lower() for c in df.columns]

y = df["charges"]
feature_cols = ["age", "sex", "bmi", "children", "smoker", "region"]
X = df[feature_cols]

cat_cols = ["sex", "smoker", "region"]
num_cols = [c for c in feature_cols if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

X_train_tr = preprocess.fit_transform(X_train)
X_test_tr = preprocess.transform(X_test)

model.fit(X_train_tr, y_train)
y_pred = model.predict(X_test_tr)

print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")

bmi_only = df[["bmi"]]
charges = df["charges"]
model_bmi = LinearRegression()
model_bmi.fit(bmi_only, charges)

plot_df = df[["bmi", "charges"]].copy().sort_values("bmi")
line_pred = model_bmi.predict(plot_df[["bmi"]])

plt.figure()
plt.scatter(df["bmi"], df["charges"], alpha=0.5)
plt.plot(plot_df["bmi"], line_pred, linewidth=3)
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.title("Charges vs BMI")
plt.tight_layout()
plt.savefig("charges_vs_bmi.png")
