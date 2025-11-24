import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('winequality-white.csv', sep=';')
print("Информация о данных:")
print(df.info())

def quality_category(q):
    if q <= 4:
        return 'плохое'
    elif q <= 6:
        return 'среднее'
    else:
        return 'хорошее'

df['quality_label'] = df['quality'].apply(quality_category)
df['quality_label'] = pd.Categorical(df['quality_label'], categories=['плохое', 'среднее', 'хорошее'])

print("\nРаспределение по категориям качества:")
print(df['quality_label'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='quality_label', palette='Set2')
plt.title('Количество вин по категориям качества')
plt.xlabel('Категория качества')
plt.ylabel('Количество')
plt.tight_layout()
plt.show()

correlation = df['fixed acidity'].corr(df['pH'])
print(f"\nКорреляция между fixed acidity и pH: {correlation:.2f}")

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='fixed acidity', y='pH', alpha=0.5)
plt.title('Зависимость между fixed acidity и pH')
plt.xlabel('Fixed Acidity')
plt.ylabel('pH')
plt.tight_layout()
plt.show()
def count_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).sum()

outlier_counts = df.select_dtypes(include='number').apply(count_outliers)
most_outliers_feature = outlier_counts.idxmax()
print(f"\nПризнак с наибольшим количеством выбросов: {most_outliers_feature}")

plt.figure(figsize=(6, 4))
sns.boxplot(data=df, y=most_outliers_feature)
plt.title(f'Ящик с усами для {most_outliers_feature}')
plt.tight_layout()
plt.show()


numeric_cols = df.select_dtypes(include='number').columns
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\nОписание стандартизированных признаков:")
print(df_scaled[numeric_cols].describe())
