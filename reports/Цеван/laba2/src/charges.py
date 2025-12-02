import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

DATA_FILE = "medical_cost_personal_dataset.csv"
IMG_FILE = "charges_vs_bmi.png"

def check_file(filename: str):
    if not os.path.isfile(filename):
        print(f"Ошибка: файл '{filename}' отсутствует.")
        sys.exit(1)

def plot_and_save(x, y, model, out_path: str):
    # Создаем точки для линии регрессии
    x_line = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
    y_line = model.predict(x_line)

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.5, label="Исходные данные")
    plt.plot(x_line, y_line, color="red", linewidth=2, label="Линия регрессии")
    plt.xlabel("BMI")
    plt.ylabel("Charges")
    plt.title("Зависимость страховых выплат от BMI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"График сохранён: {out_path}")

def main():
    check_file(DATA_FILE)

    data = pd.read_csv(DATA_FILE)
    print("Файл загружен:", DATA_FILE, "| размер:", data.shape)

    if {"bmi", "charges"} - set(data.columns):
        print("Ошибка: столбцы 'bmi' и/или 'charges' отсутствуют.")
        sys.exit(1)

    # Берём только нужные данные
    subset = data[["bmi", "charges"]].dropna()
    if subset.shape[0] < 2:
        print("Недостаточно данных для построения графика.")
        sys.exit(1)

    X = subset[["bmi"]].values
    y = subset["charges"].values

    # Линейная регрессия
    model = LinearRegression()
    model.fit(X, y)

    plot_and_save(X, y, model, IMG_FILE)
    print("Готово.")

if __name__ == "__main__":
    main()
