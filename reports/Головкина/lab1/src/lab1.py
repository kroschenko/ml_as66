import io
from typing import Optional
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

def main() -> None:
    # локальный путь к датасету
    local_path = Path(r"C:\Users\Пипка\Documents\omo\Titanic-Dataset.csv")

    # Попытка загрузить из локального файла; при ошибке — fallback на URL
    if local_path.exists():
        print(f"Загружаю данные из локального файла: {local_path}")
        try:
            df = pd.read_csv(local_path)
        except Exception as e:
            print(f"Ошибка при чтении локального CSV: {e}")
            df = None
    else:
        print(f"Локальный файл не найден: {local_path}")
        df = None

    if df is None:
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        print(f"Загружаю данные из удалённого URL: {url}")
        df = pd.read_csv(url)

    # 1) Загрузка данных и вывод первых 5 строк + .info()
    print("Первые 5 строк:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(df.head(5))

    print("\nИнформация о столбцах (.info()):")
    buf = io.StringIO()
    df.info(buf=buf)
    print(buf.getvalue())

    # 2) Столбчатая диаграмма: выжившие/погибшие
    plt.figure(figsize=(6, 4))
    survived_counts = df["Survived"].value_counts().reindex([0, 1], fill_value=0)
    survived_counts.index = ["Мертвы (0)", "Живы (1)"]
    survived_counts.plot(kind="bar", color=["#d9534f", "#5cb85c"])
    plt.title("Количество погибших и выживших")
    plt.ylabel("Число пассажиров")
    plt.tight_layout()
    plt.show()

    # 3) Заполнение пропусков в Age медианой
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    age_median = df["Age"].median()
    df["Age"] = df["Age"].fillna(age_median)
    print(f"\nПропуски в Age заполнены медианой: {age_median:.2f}")

    # 4) One-Hot Encoding для Sex и Embarked
    df_ohe = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=False)
    created_ohe = [c for c in df_ohe.columns if c.startswith("Sex_") or c.startswith("Embarked_")]
    print("\nСозданы OHE-признаки:", ", ".join(created_ohe))

    # 5) Гистограмма распределения возрастов
    plt.figure(figsize=(6, 4))
    plt.hist(df_ohe["Age"], bins=30, color="#5bc0de", edgecolor="white")
    plt.title("Распределение возраста пассажиров (Age)")
    plt.xlabel("Возраст")
    plt.ylabel("Частота")
    plt.tight_layout()
    plt.show()

    # 6) Признак FamilySize = SibSp + Parch
    df_ohe["FamilySize"] = df_ohe["SibSp"].fillna(0) + df_ohe["Parch"].fillna(0)
    print("\nПризнак FamilySize создан. Примеры значений:")
    print(df_ohe[["SibSp", "Parch", "FamilySize"]].head())

if __name__ == "__main__":
    main()