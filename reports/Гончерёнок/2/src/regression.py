import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

df_medical = pd.read_csv('medical_cost_personal_dataset.csv')

categorical_columns = ['sex', 'smoker', 'region']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df_medical[col + '_encoded'] = le.fit_transform(df_medical[col])
    label_encoders[col] = le

X_reg = df_medical[['age', 'bmi', 'children', 'sex_encoded', 'smoker_encoded', 'region_encoded']]
y_reg = df_medical['charges']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train_reg, y_train_reg)

y_pred_reg = lr_model.predict(X_test_reg)

mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print("РЕГРЕССИЯ  МЕДИЦИНСКИЕ РАСХОДЫ  regr.py:34 - regression.py:34")
print(f"MAE: {mae:.2f}  regr.py:35 - regression.py:35")
print(f"R² Score: {r2:.4f}  regr.py:36 - regression.py:36")

plt.figure(figsize=(10, 6))
plt.scatter(df_medical['bmi'], df_medical['charges'], alpha=0.6, color='blue')
z = np.polyfit(df_medical['bmi'], df_medical['charges'], 1)
p = np.poly1d(z)
plt.plot(df_medical['bmi'], p(df_medical['bmi']), "r--", alpha=0.8, linewidth=2)
plt.xlabel('BMI (Индекс массы тела)')
plt.ylabel('Медицинские расходы (charges)')
plt.title('Зависимость расходов от BMI')
plt.grid(True, alpha=0.3)
plt.savefig('medical_costs_regression.png', dpi=300, bbox_inches='tight')
plt.show()

feature_importance = pd.DataFrame({
    'Feature': X_reg.columns,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)
print("\nВажность признаков:  regr.py:54 - regression.py:54")
print(feature_importance)