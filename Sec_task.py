import pandas as pd
from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['median_house_value'] = california.target

# Оставляем только нужные столбцы: Longitude, Latitude, HouseAge
df_trimmed = df[['Longitude', 'Latitude', 'HouseAge']]

# Выводим первые 10 строк результата
print("Таблица после удаления лишних столбцов:")
print(df_trimmed.head(10))

# Выводим информацию о размере таблицы
print("\nРазмер таблицы:")
print(f"Количество строк и столбцов: {df_trimmed.shape}")