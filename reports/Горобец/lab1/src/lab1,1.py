import pandas as pd

df = pd.read_csv(r'C:\Users\Anton\Downloads\iris.csv')

pd.set_option('display.max_rows', None)     # Показывать все строки
pd.set_option('display.max_columns', None)  # Показывать все столбцы
pd.set_option('display.width', None)        # Без ограничения по ширине
pd.set_option('display.max_colwidth', None) # Полная ширина столбцов

print(df)

print("\nПроверка на пропущенные значения:")
print(df.isnull().sum())