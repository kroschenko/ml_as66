import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv(r'C:\Users\Anton\Downloads\iris.csv')

# Настройки отображения
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Построение box plot для признака petal.length
df.boxplot(column='petal.length', by='variety', grid=False)

plt.title('Распределение petal.length по видам ирисов')
plt.suptitle('')  # Убираем автоматический заголовок
plt.xlabel('Вид ириса')
plt.ylabel('Petal Length (cm)')
plt.show()
