import pandas as pd
df = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\1лаба ОМО\\heart.csv')
df['sex']=df['sex'].map({0: 'female', 1: 'male'})
df_=pd.get_dummies(df, columns=['sex'])
df_=df_.astype(int)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
print(df_)