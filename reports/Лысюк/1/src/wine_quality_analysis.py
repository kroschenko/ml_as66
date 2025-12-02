import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

file_path = 'winequality-red.csv'
df = pd.read_csv(file_path, sep=';')
print('Информация о данных:')
print(df.info())


def quality_to_category(q):
    if q <= 4:
        return 'плохое'
    elif 5 <= q <= 6:
        return 'среднее'
    else:
        return 'хорошее'


df['quality_cat'] = df['quality'].apply(quality_to_category)
print('\nРаспределение по категориям качества:')
print(df['quality_cat'].value_counts())

df['quality_cat'].value_counts().plot(
    kind='bar', color=['red', 'orange', 'green'])
plt.title('Количество вин по категориям качества')
plt.xlabel('Категория качества')
plt.ylabel('Количество')
plt.show()

corr = df['fixed acidity'].corr(df['pH'])
print(f'\nКорреляция fixed acidity и pH: {corr:.3f}')

plt.scatter(df['fixed acidity'], df['pH'], alpha=0.5)
plt.title('Диаграмма рассеяния fixed acidity vs pH')
plt.xlabel('fixed acidity')
plt.ylabel('pH')
plt.show()


def count_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return len(outliers)


numeric_cols = df.select_dtypes(include=np.number).columns.drop('quality')
outliers_counts = {col: count_outliers(df[col]) for col in numeric_cols}
max_outliers_feature = max(outliers_counts, key=outliers_counts.get)

print(
    f'\nПризнак с наибольшим количеством выбросов: {max_outliers_feature} ({outliers_counts[max_outliers_feature]} выбросов)')

plt.boxplot(df[max_outliers_feature])
plt.title(f'Box plot для {max_outliers_feature}')
plt.show()

scaler = StandardScaler()
features_to_scale = numeric_cols
df_scaled = df.copy()
df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])

print('\nПример стандартизированных данных:')
print(df_scaled.head())
