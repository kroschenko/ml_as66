import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# 1. Загрузка данных и вывод информации
df = pd.read_csv("E:/Projects/ml_as66/reports/Nerush/lab1/src/german_credit.csv")
print("🔹 Информация о данных:")
print(df.info())

# 2. Анализ и визуализация целей кредита
purpose_counts = df['purpose'].value_counts().head(5)
plt.figure(figsize=(8, 5))
purpose_df = purpose_counts.reset_index()
purpose_df.columns = ['purpose', 'count']
sns.barplot(data=purpose_df, x='purpose', y='count', hue='purpose', palette="Blues", legend=False)
plt.title("Топ-5 целей кредита")
plt.ylabel("Количество")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Преобразование категориальных признаков Sex и Housing
df_encoded = pd.get_dummies(df, columns=['personal_status_sex', 'housing'], drop_first=True)
print(df_encoded.head())

# 4. Ящик с усами для Credit amount по default
plt.figure(figsize=(8, 5))
sns.boxplot(x='default', y='credit_amount', hue='default', data=df, palette="Set2", legend=False)
plt.title("Сравнение суммы кредита по кредитоспособности")
plt.xlabel("Кредитоспособность (0 = плохой, 1 = хороший)")
plt.ylabel("Сумма кредита")
plt.tight_layout()
plt.show()

# 5. Сводная таблица по Credit history
pivot_table = df.pivot_table(
    values=['age', 'duration_in_month'],
    index='credit_history',
    aggfunc='mean'
)
print("\n🔹 Средний возраст и длительность кредита по кредитной истории:")
print(pivot_table)

# 6. Нормализация числовых признаков
scaler = MinMaxScaler()
num_cols = ['age', 'credit_amount', 'duration_in_month']
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])
print(df_encoded[num_cols].head())
