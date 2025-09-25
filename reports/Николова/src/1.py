import pandas as pd 
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
df = pd.read_csv('C:\\Users\\z3594\\OneDrive\\Документы\\лабы\\heart.csv')
print(df)
print("Есть ли пропуски: ", df.isnull().values.any())