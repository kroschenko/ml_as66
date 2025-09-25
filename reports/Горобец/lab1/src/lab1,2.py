import pandas as pd

df = pd.read_csv(r'C:\Users\Anton\Downloads\iris.csv')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("\nКоличество образцов каждого вида ириса:")
print(df['variety'].value_counts())
