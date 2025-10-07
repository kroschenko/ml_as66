import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1.  Загрузка данных по конкретной ссылке.  Для этого требуется установка gdown и наличие файла по ссылке.
#     Иначе, переключаемся на встроенный в sklearn датасет.
try:
    import gdown

    file_id = '1Hw1xVRsIj-6PzOhOv825IUIWsRSPJR8m'

    url = f'https://drive.google.com/uc?id={file_id}'

    output = 'california_housing.csv'

    # Скачайть файл.
    gdown.download(url, output, quiet=False)  # quiet=False покажет процесс загрузки

    # Загрузите данные из CSV-файла.
    df = pd.read_csv(output)

    print("Данные успешно загружены из Google Drive.")

except ImportError:
    print("Библиотека gdown не установлена.  Пожалуйста, установите её с помощью pip install gdown.")
    print("Переключение на встроенный датасет fetch_california_housing.")
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=True)
    df = data.frame

except Exception as e:
    print(f"Ошибка при загрузке данных из Google Drive: {e}")
    print("Переключение на встроенный датасет fetch_california_housing.")
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=True)
    df = data.frame


# 2. Подготовка данных (независимо от источника)
if 'MedHouseVal' in df.columns:
    X = df.drop(columns='MedHouseVal')
    y = df['MedHouseVal']
else:  #  Если вдруг назван иначе, то ошибка
    print("Столбец 'MedHouseVal' не найден.  Убедитесь, что CSV-файл содержит этот столбец.")
    raise ValueError("Не найден столбец 'MedHouseVal'")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Предсказания
y_pred = model.predict(X_test)

# 5. Оценка качества
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")


# 6. Визуализация зависимости от median_income
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test['MedInc'], y=y_test, label='Фактические значения')

# Правильное построение линии регрессии
medinc_test = X_test['MedInc']
y_pred_test = model.predict(X_test)

plot_data = pd.DataFrame({'MedInc': medinc_test, 'Predicted': y_pred_test})
plot_data = plot_data.sort_values('MedInc')

sns.lineplot(x=plot_data['MedInc'], y=plot_data['Predicted'], color='red', label='Линия регрессии')

plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Зависимость стоимости жилья от дохода')
plt.legend()
plt.tight_layout()
plt.show()