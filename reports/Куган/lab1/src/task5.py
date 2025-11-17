import pandas as pd
df = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\1лаба ОМО\\heart.csv')
chol_diseased = df[df['target'] == 1]['chol'].mean()
chol_healthy = df[df['target'] == 0]['chol'].mean()
print(f"Средний уровень холестерина у больных пациентов: {chol_diseased:.2f}")
print(f"Средний уровень холестерина у здоровых пациентов: {chol_healthy:.2f}")