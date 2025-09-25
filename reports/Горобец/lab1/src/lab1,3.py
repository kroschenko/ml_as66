import pandas as pd
import matplotlib.pyplot as plt
import itertools

# Загрузка данных
df = pd.read_csv(r'C:\Users\Anton\Downloads\iris.csv')

# Настройки отображения
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Выделение признаков и классов
features = df.drop('variety', axis=1).columns
classes = df['variety'].unique()
colors = ['red', 'green', 'blue']
color_map = dict(zip(classes, colors))

# Построение парных диаграмм
fig, axes = plt.subplots(len(features), len(features), figsize=(12, 12))
for i, x_feature in enumerate(features):
    for j, y_feature in enumerate(features):
        ax = axes[i, j]
        for variety in classes:
            subset = df[df['variety'] == variety]
            if i == j:
                ax.hist(subset[x_feature], color=color_map[variety], alpha=0.5, label=variety)
            else:
                ax.scatter(subset[y_feature], subset[x_feature], color=color_map[variety], alpha=0.6, label=variety)
        if i == len(features) - 1:
            ax.set_xlabel(y_feature)
        else:
            ax.set_xticks([])
        if j == 0:
            ax.set_ylabel(x_feature)
        else:
            ax.set_yticks([])

# Легенда
handles = [plt.Line2D([0], [0], marker='o', color='w', label=variety,
                      markerfacecolor=color_map[variety], markersize=8) for variety in classes]
fig.legend(handles=handles, loc='upper right', title='Вид ириса')

plt.suptitle("Парные диаграммы рассеяния с цветами по видам", fontsize=16)
plt.tight_layout()
plt.show()
