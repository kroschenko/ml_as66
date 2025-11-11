import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\1лаба ОМО\\heart.csv')

plt.scatter(df['age'], df['thalach'], c=df['target'], cmap='coolwarm')

plt.xlabel('Возраст')
plt.ylabel('Максимальный пульс')
plt.show(block=True)