# analyze_german_credit.py
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

sns.set(style="whitegrid", font_scale=1.05)

# --- Настройки ---
DATA_PATH = "german_credit.csv"  # <- поменяй путь/имя файла при необходимости
OUT_DIR = "german_credit_output"
os.makedirs(OUT_DIR, exist_ok=True)


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл {path} не найден. Помести CSV/Excel файл и обнови DATA_PATH.")

    ext = os.path.splitext(path)[1].lower()

    if ext in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
        # если получилась одна колонка с запятыми — пробуем перечитать как CSV
        if df.shape[1] == 1:
            first_col = df.columns[0]
            if isinstance(first_col, str) and "," in first_col:
                print("Похоже, это CSV внутри Excel — пробуем читать как CSV")
                try:
                    df = pd.read_csv(path)
                except Exception as e:
                    print("Не удалось прочитать как CSV, оставляем Excel-версию:", e)
    else:
        df = pd.read_csv(path)

    return df


def infer_and_prepare_columns(df):
    # Показываем колонки
    print("Колонки в наборе данных:\n", list(df.columns), "\n")

    colmap = {}
    cols = df.columns.str.lower()

    # Sex
    if "sex" in df.columns:
        colmap["sex"] = "Sex"
    else:
        cand = [c for c in df.columns if "personal" in c.lower() and "sex" in c.lower() or "personal status" in c.lower()]
        if cand:
            colmap["sex"] = cand[0]
        else:
            for c in df.columns:
                vals = df[c].dropna().astype(str).str.lower()
                if vals.isin(["male", "female", "m", "f", "man", "woman"]).any():
                    colmap["sex"] = c
                    break

    # Housing
    for name in ["Housing", "housing", "house"]:
        if name in df.columns:
            colmap["housing"] = name
            break
    if "housing" not in colmap:
        cand = [c for c in df.columns if "housing" in c.lower() or "home" in c.lower()]
        if cand:
            colmap["housing"] = cand[0]

    # Risk
    for target in ["Risk", "risk", "class", "target", "credit_risk", "creditability"]:
        if target in df.columns:
            colmap["risk"] = target
            break
    if "risk" not in colmap:
        for c in df.columns:
            vals = df[c].dropna().astype(str).str.lower().unique()
            if set(vals).intersection({"good", "bad", "g", "b"}):
                colmap["risk"] = c
                break

    # Purpose
    for name in ["Purpose", "purpose", "purpose of loan"]:
        if name in df.columns:
            colmap["purpose"] = name
            break
    if "purpose" not in colmap:
        cand = [c for c in df.columns if "purpose" in c.lower() or "use" in c.lower()]
        if cand:
            colmap["purpose"] = cand[0]

    # Credit amount
    for name in ["Credit amount", "CreditAmount", "credit amount", "credit_amount", "amount"]:
        if name in df.columns:
            colmap["credit_amount"] = name
            break
    if "credit_amount" not in colmap:
        cand = [c for c in df.columns if "credit" in c.lower() and "amount" in c.lower()]
        if cand:
            colmap["credit_amount"] = cand[0]

    # Age
    for name in ["age", "customer_age"]:
        if name in df.columns:
            colmap["age"] = name
            break

    # Duration
    for name in ["Duration", "duration", "duration in month", "duration_month"]:
        if name in df.columns:
            colmap["duration"] = name
            break
    if "duration" not in colmap:
        cand = [c for c in df.columns if "duration" in c.lower() or "month" in c.lower() and "duration" in c.lower()]
        if cand:
            colmap["duration"] = cand[0]

    # Credit history
    for name in ["Credit history", "CreditHistory", "credit history", "credit_history"]:
        if name in df.columns:
            colmap["credit_history"] = name
            break
    if "credit_history" not in colmap:
        cand = [c for c in df.columns if "credit" in c.lower() and "history" in c.lower()]
        if cand:
            colmap["credit_history"] = cand[0]

    print("Найденные важные колонки (по возможности):", colmap, "\n")
    return colmap


def exploratory_info(df):
    print("=== Общая информация ===")
    print(df.info())
    print("\n--- Количество пропусков по столбцам ---")
    print(df.isnull().sum())
    print("\n--- Основные статистические показатели для числовых столбцов ---")
    try:
        print(df.describe().T[['mean', '50%', 'std']].rename(columns={"50%": "median"}))
    except Exception:
        print("Не удалось вывести describe() — возможно, нет числовых колонок.")
    print("\n")


def handle_missing(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in num_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mean())
    for c in cat_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    return df


def extract_sex(df, sex_col_name):
    if sex_col_name not in df.columns:
        return df
    sample = df[sex_col_name].astype(str).str.lower()
    if sample.isin(["male", "female", "m", "f", "man", "woman"]).any():
        def norm(x):
            x = str(x).lower()
            if "male" in x or x in ("m", "man"):
                return "male"
            if "female" in x or x in ("f", "woman"):
                return "female"
            return x
        df["Sex_extracted"] = df[sex_col_name].apply(norm)
    else:
        def find_mf(x):
            s = str(x).lower()
            if "male" in s or "m" in s.split():
                return "male"
            if "female" in s or "f" in s.split():
                return "female"
            return s
        df["Sex_extracted"] = df[sex_col_name].apply(find_mf)
    return df


def encode_sex_housing(df, colmap):
    if "sex" in colmap:
        df = extract_sex(df, colmap["sex"])
        sex_col = "Sex_extracted" if "Sex_extracted" in df.columns else colmap["sex"]
        df["Sex_male"] = df[sex_col].astype(str).str.lower().map(lambda x: 1 if "male" in x else 0)
    else:
        print("Колонка пола не найдена — пропускаем кодирование Sex.")

    if "housing" in colmap:
        housing_col = colmap["housing"]
        df[housing_col] = df[housing_col].astype(str)
        d = pd.get_dummies(df[housing_col], prefix="Housing", drop_first=True)
        df = pd.concat([df, d], axis=1)
    else:
        print("Колонка Housing не найдена — пропускаем кодирование Housing.")

    return df


def plot_top5_purpose(df, purpose_col):
    if purpose_col not in df.columns:
        print("Не найдена колонка Purpose — пропускаем анализ распределения целей.")
        return
    counts = df[purpose_col].astype(str).value_counts().head(5)
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=counts.values, y=counts.index)
    ax.set_xlabel("Количество")
    ax.set_ylabel("Цель кредита")
    ax.set_title("Топ-5 целей кредита (Purpose)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "top5_purpose.png"))
    plt.close()
    print("График top5_purpose.png сохранён в", OUT_DIR)


def boxplot_credit_by_risk(df, credit_col, risk_col):
    if credit_col not in df.columns or risk_col not in df.columns:
        print("Отсутствует колонка для boxplot (Credit amount или Risk). Пропускаем.")
        return
    plt.figure(figsize=(8, 5))
    ax = sns.boxplot(x=df[risk_col].astype(str), y=df[credit_col])
    ax.set_xlabel("Кредитоспособность (Risk)")
    ax.set_ylabel("Сумма кредита")
    ax.set_title("Boxplot: Credit amount по Risk")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "boxplot_credit_by_risk.png"))
    plt.close()
    print("График boxplot_credit_by_risk.png сохранён в", OUT_DIR)


def pivot_age_duration_by_credit_history(df, credit_history_col):
    if credit_history_col not in df.columns:
        print("Колонка Credit history не найдена — пропускаем сводную таблицу.")
        return None
    pivot = df.pivot_table(values=["age", "duration_in_month"], index=credit_history_col, aggfunc="mean")
    print("\nСводная таблица: средний Age и Duration по Credit history:\n")
    print(pivot)
    pivot.to_csv(os.path.join(OUT_DIR, "pivot_age_duration_by_credit_history.csv"))
    print("\nPivot сохранён в", os.path.join(OUT_DIR, "pivot_age_duration_by_credit_history.csv"))
    return pivot


def normalize_numeric_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print("Невозможно нормализовать, т.к. не найдены столбцы:", missing)
    present = [c for c in cols if c in df.columns]
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[present] = scaler.fit_transform(df_norm[present])
    df_norm[present].to_csv(os.path.join(OUT_DIR, "normalized_numeric_columns.csv"), index=False)
    print("\nНормализованные колонки сохранены в", os.path.join(OUT_DIR, "normalized_numeric_columns.csv"))
    return df_norm


def main():
    df = load_data(DATA_PATH)
    colmap = infer_and_prepare_columns(df)
    exploratory_info(df)

    df = handle_missing(df)
    print("Пропуски обработаны (числовые -> среднее, категориальные -> мода).\n")

    plot_top5_purpose(df, colmap.get("purpose"))
    df = encode_sex_housing(df, colmap)
    boxplot_credit_by_risk(df, "credit_amount", "default")
    pivot_age_duration_by_credit_history(df, colmap.get("credit_history"))

    numeric_to_norm = []
    for name in ["age", "credit_amount", "duration"]:
        if name in colmap:
            numeric_to_norm.append(colmap[name])
    for cand in ["age", "Credit amount", "CreditAmount", "credit amount", "Duration", "duration"]:
        if cand in df.columns and cand not in numeric_to_norm:
            low = cand.lower()
            if "age" in low or ("credit" in low and "amount" in low) or "duration" in low:
                numeric_to_norm.append(cand)
    numeric_to_norm = list(dict.fromkeys(numeric_to_norm))
    print("Колонки, которые будут нормализованы:", numeric_to_norm)
    df_norm = normalize_numeric_columns(df, numeric_to_norm)

    out_csv = os.path.join(OUT_DIR, "german_credit_processed.csv")
    df_norm.to_csv(out_csv, index=False)
    print("\nПолный обработанный датасет сохранён:", out_csv)

    print("\nГотово. Проверь файлы в папке", OUT_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Ошибка при выполнении:", e)
        sys.exit(1)
