from pandas import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class BostonHousingAnalyze:
    def __init__(self, dataset_path: str = "BostonHousing.csv"):
        self.boston = read_csv(dataset_path)
        self.correlation_matrix = self.boston.corr()
        self.medv_correlations = self.correlation_matrix['MEDV'].abs().sort_values(ascending=False)
        self.most_correlated_feature = self.medv_correlations.index[1]
        self.most_correlated_value = self.medv_correlations[1]
        self.scaler = MinMaxScaler()
        self.numeric_columns = self.boston.select_dtypes(include=[np.number]).columns
        self.boston_normalized = self.boston.copy()
        self.boston_normalized[self.numeric_columns] = self.scaler.fit_transform(self.boston[self.numeric_columns])

    @staticmethod
    def beautify(func):
        def wrapper(*args, **kwargs):
            print(f"\nВыполнение: {func.__name__}\n" + "=" * 80 + "\n")
            result = func(*args, **kwargs)
            print("\n" + "=" * 80 + "\n")
            return result

        return wrapper

    @beautify
    def zad_1(self):
        print(f"Основные статистические характеристики:\n{self.boston.describe()}")

    @beautify
    def zad_2(self):
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Матрица корреляции Boston Housing')
        plt.tight_layout()
        plt.show()

    @beautify
    def zad_3(self):
        print(f"Признак, наиболее сильно коррелирующий с MEDV: {self.most_correlated_feature}")
        print(f"Коэффициент корреляции: {self.most_correlated_value:.3f}")
        print(f"Корреляции всех признаков с MEDV:\n{self.medv_correlations}")

    @beautify
    def zad_4(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.boston[self.most_correlated_feature], self.boston['MEDV'], alpha=0.6)
        plt.xlabel(self.most_correlated_feature.upper())
        plt.ylabel('MEDV (Медианная стоимость)')
        plt.title(f'Диаграмма рассеяния: {self.most_correlated_feature.upper()} vs MEDV')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @beautify
    def zad_5(self):
        print("Данные после нормализации (первые 5 строк):")
        print(self.boston_normalized.head())
        print("\nПроверка диапазона (min/max) после нормализации:")
        print(self.boston_normalized[self.numeric_columns].agg(['min', 'max']))

    @beautify
    def zad_6(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(self.boston['CRIM'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Уровень преступности (CRIM)')
        plt.ylabel('Частота')
        plt.title('Распределение CRIM (оригинальные данные)')
        plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.hist(self.boston_normalized['CRIM'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.xlabel('Уровень преступности (CRIM) нормализованный')
        plt.ylabel('Частота')
        plt.title('Распределение CRIM (после нормализации)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def __call__(self):
        self.zad_1()
        self.zad_2()
        self.zad_3()
        self.zad_4()
        self.zad_5()
        self.zad_6()


if __name__ == "__main__":
    BostonHousingAnalyze()()
