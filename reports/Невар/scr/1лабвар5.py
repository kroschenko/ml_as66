import pandas as pd
import matplotlib.pyplot as plt

# url для набора данных Adult Census Income из репозитория uci
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

# имена столбцов согласно описанию набора данных
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]

# загружаем данные в DataFrame
try:
    df = pd.read_csv(
        url,
        header=None,
        names=column_names,
        sep=',\s',
        na_values='?',
        engine='python'
    )
    print("Данные успешно загружены.")
    data_loaded_successfully = True
except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")
    data_loaded_successfully = False

if data_loaded_successfully:
    # ЗАДАЧА 1: загрузите данные и выведите первые 10 строк
    print("\nЗадача 1: Первые 10 строк набора данных")
    print(df.head(10))

    print("\nИсследовательский анализ")
    print("\nИнформация о DataFrame:")
    df.info()

    print("\nСтатистические показатели для числовых столбцов:")
    print(df.describe())

    print("\nКоличество пропущенных значений в каждом столбце:")
    print(df.isnull().sum())

    # ЗАДАЧА 2: проанализируйте столбец workclass. Найдите и замените значения '?' на наиболее часто встречающееся значение
    print("\nЗадача 2: Обработка пропусков в 'workclass'")
    print("\nРаспределение значений в 'workclass' до обработки:")
    print(df['workclass'].value_counts())

    workclass_mode = df['workclass'].mode()[0]
    print(f"\nНаиболее частое значение (мода) для 'workclass': '{workclass_mode}'")

    df['workclass'].fillna(workclass_mode, inplace=True)
    print("\nПропущенные значения в 'workclass' заменены.")

    print("\nРаспределение значений в 'workclass' после обработки:")
    print(df['workclass'].value_counts())
    print(f"\nКоличество пропусков в 'workclass' после замены: {df['workclass'].isnull().sum()}")

    # ЗАДАЧА 3: определите, сколько в наборе данных мужчин и женщин. Визуализируйте результат
    print("\nЗадача 3: Распределение по полу")
    gender_counts = df['sex'].value_counts()
    print("Количество мужчин и женщин:")
    print(gender_counts)

    plt.figure(figsize=(8, 6))
    bars = plt.bar(gender_counts.index, gender_counts.values, color=['lightblue', 'lightpink'])
    plt.title('Распределение по полу в наборе данных', fontsize=16)
    plt.xlabel('Пол', fontsize=12)
    plt.ylabel('Количество', fontsize=12)
    
    for bar, value in zip(bars, gender_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                str(value), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    print("\nВизуализация распределения по полу создана.")

    # ЗАДАЧА 4: преобразуйте категориальный признак race в числовой формат
    print("\nЗадача 4: Преобразование 'race' в числовой формат (One-Hot Encoding)")
    print("\nПервые 5 значений столбца 'race' до преобразования:")
    print(df['race'].head())

    df_encoded = pd.get_dummies(df, columns=['race'], prefix='race')

    print("\nDataFrame после One-Hot Encoding (показаны новые столбцы 'race_*'):")
    race_columns = [col for col in df_encoded.columns if 'race_' in col]
    print(df_encoded[['age'] + race_columns].head())

    # ЗАДАЧА 5: постройте гистограмму распределения возраста (age) для двух групп
    print("\nЗадача 5: Гистограмма распределения возраста по уровню дохода")

    plt.figure(figsize=(12, 7))
    
    age_low_income = df[df['income'] == '<=50K']['age']
    age_high_income = df[df['income'] == '>50K']['age']
    
    plt.hist(age_low_income, bins=30, alpha=0.7, color='blue', label='<=50K', edgecolor='black')
    plt.hist(age_high_income, bins=30, alpha=0.7, color='red', label='>50K', edgecolor='black')

    plt.title('Распределение возраста по уровню дохода', fontsize=16)
    plt.xlabel('Возраст', fontsize=12)
    plt.ylabel('Частота', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.show()
    print("\nГистограмма распределения возраста создана.")

    # ЗАДАЧА 6: создайте новый бинарный признак is_usa на основе столбца native-country
    print("\nЗадача 6: Создание бинарного признака 'is_usa'")

    df_encoded['is_usa'] = (df_encoded['native-country'] == 'United-States').astype(int)

    print("\nПримеры нового столбца 'is_usa' и исходного 'native-country':")
    print(df_encoded[['native-country', 'is_usa']].tail(10))

    print("\nРаспределение значений в новом столбце 'is_usa':")
    print(df_encoded['is_usa'].value_counts())