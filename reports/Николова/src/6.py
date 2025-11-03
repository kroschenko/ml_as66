import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('C:\\Users\\z3594\\OneDrive\\Документы\\лабы\\heart.csv')

features_to_normalize = ['age', 'trestbps', 'chol', 'thalach']

scaler = MinMaxScaler()

df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print(df)