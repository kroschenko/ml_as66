import pandas as pd
import matplotlib.pyplot as plt

# 1.
df = pd.read_csv('Melbourne_housing.csv', na_values=['missing', 'inf'],)

missing_counts = df.isna().sum()
print("Количество пропусков по столбцам:")
print(missing_counts.sort_values(ascending=False).head(5))

most_missing = missing_counts.idxmax()
print(f"\nУдаляем столбец с наибольшим количеством пропусков: {most_missing} ({missing_counts[most_missing]} пропусков)")
df.drop(columns=[most_missing], inplace=True)

print(f"Оставшиеся столбцы: {list(df.columns)}")



# 2
before_rows = len(df)
df.dropna(subset=['Price'], inplace=True)
after_rows = len(df)

print(f"\nУдалено строк без цены: {before_rows - after_rows}")
print(f"Оставшиеся строки: {after_rows}")



# 3
plt.figure(figsize=(10, 6))
plt.hist(df['Price'], bins=40, color='skyblue', edgecolor='black')
plt.title('Распределение цен на недвижимость в Мельбурне')
plt.xlabel('Цена (AUD)')
plt.ylabel('Количество объектов')
plt.grid(True)
plt.tight_layout()
plt.show()



# 4
top_suburbs = df['Suburb'].value_counts().nlargest(5).index
avg_prices = df[df['Suburb'].isin(top_suburbs)].groupby('Suburb')['Price'].mean()
print("\nСредняя цена по 5 самым популярным пригородам:")
print(avg_prices)



# 5
current_year = pd.Timestamp.now().year
df['PropertyAge'] = current_year - pd.to_numeric(df['YearBuilt'])

print("\nШапка DataFrame с PropertyAge:")
print(df[['Suburb', 'YearBuilt', 'PropertyAge']].head())



# 6
df = pd.get_dummies(df, columns=['Type'], prefix='Type')

print("\nНовые столбцы после One-Hot Encoding:")
print([col for col in df.columns if col.startswith('Type_')])

print("\nШапка итогового DataFrame:")
print(df.head())
