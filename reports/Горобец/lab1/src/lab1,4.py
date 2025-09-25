import pandas as pd
df = pd.read_csv(r'C:\Users\Anton\Downloads\iris.csv')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
mean_by_variety = df.groupby('variety').mean(numeric_only=True)
print("\nСредние значения признаков по каждому виду ириса:")
print(mean_by_variety)
