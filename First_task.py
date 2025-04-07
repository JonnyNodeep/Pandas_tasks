import pandas as pd
from sklearn.datasets import fetch_california_housing

# Загружаем данные из sklearn
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['median_house_value'] = california.target
print("1. Последние 5 строк таблицы:")
print(df.tail())

print("\n2. Размер таблицы:")

print(f"Количество строк и столбцов: {df.shape}")

print("\n3. Каждая вторая строка:")