'''
Задача: Загрузите таблицу в переменную df по указанному пути 
('/content/sample_data/california_housing_train.csv'). Выведите информацию по столбцам:
Выведите последние 5 строк
Выведите размер таблицы
Выведите каждую 2 строку
'''

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
print(df[::2].head())  # Показываем первые несколько строк для наглядности