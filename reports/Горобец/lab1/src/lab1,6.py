import pandas as pd
from sklearn.preprocessing import StandardScaler

# Загрузка данных
df = pd.read_csv(r'C:\Users\Anton\Downloads\iris.csv')

# Настройки отображения
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Выделение числовых признаков
numeric_columns = df.select_dtypes(include='number').columns

# Стандартизация
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Вывод стандартизированных данных
print("\nСтандартизированные данные:")
print(df)
