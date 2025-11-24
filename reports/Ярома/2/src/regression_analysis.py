import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

MED_CSV = "medical_cost_personal_dataset.csv"
OUT_PNG = "charges_vs_bmi.png"
RANDOM_STATE = 42

def ensure_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

def save_figure(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"PNG сохранён: {path}")

def main():
    try:
        ensure_exists(MED_CSV)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    df = pd.read_csv(MED_CSV)
    print("Загружен файл:", MED_CSV, "| shape:", df.shape)

    # Проверка обязательных столбцов
    if 'charges' not in df.columns:
        print("В датасете отсутствует столбец charges")
        sys.exit(0)

    if 'bmi' not in df.columns:
        print("В датасете отсутствует столбец bmi")
        sys.exit(0)


    plot_df = df[['bmi', 'charges']].dropna()
    if plot_df.shape[0] < 2:
        print("Недостаточно данных с обеими колонками 'bmi' и 'charges' для построения графика — ничего не делаю.")
        sys.exit(0)

    X_bmi = plot_df[['bmi']].values
    y_charges = plot_df['charges'].values

    # Простая линейная регрессия
    lr = LinearRegression()
    lr.fit(X_bmi, y_charges)
    xs = np.linspace(X_bmi.min(), X_bmi.max(), 200).reshape(-1, 1)
    ys = lr.predict(xs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_bmi, y_charges, alpha=0.6, s=25, label="data (bmi and charges)")
    ax.plot(xs, ys, linewidth=2, label=f"linear fit (slope={lr.coef_[0]:.2f})")
    ax.set_xlabel("bmi")
    ax.set_ylabel("charges")
    ax.set_title("Charges and BMI")
    ax.legend()

    save_figure(fig, OUT_PNG)
    print("График построен и сохранён. Завершение.")

if __name__ == "__main__":
    main()
