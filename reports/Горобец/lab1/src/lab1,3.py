import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

df = pd.read_csv(r'C:\Users\Anton\Downloads\iris.csv')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

scatter_matrix(df.drop('variety', axis=1), figsize=(10, 10), diagonal='hist', marker='o', alpha=0.8)

plt.suptitle("Парные диаграммы рассеяния для признаков Iris", fontsize=14)
plt.show()
