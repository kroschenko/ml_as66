import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('C:\\Users\\z3594\\OneDrive\\Документы\\лабы\\heart.csv')

counts = df['target'].value_counts()

plt.bar(counts.index, counts.values, color='lightcoral', edgecolor='black')
plt.xticks([0,1],['Больные','Здоровые'])
plt.grid(axis='y')
plt.show(block=True)