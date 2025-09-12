import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# 1. Генерация синтетических данных
np.random.seed(42)  # Для воспроизводимости
n_transactions = 1000

# Создаем данные с русскоязычными столбцами
data = {
    'Идентификатор_счета': [f'СЧЕТ-{i:04d}' for i in range(1, n_transactions + 1)],
    'Филиал': np.random.choice(['А', 'Б', 'В'], n_transactions),
    'Город': np.random.choice(['Москва', 'Санкт-Петербург', 'Казань'], n_transactions),
    'Тип_клиента': np.random.choice(['Постоянный', 'Обычный'], n_transactions),
    'Пол': np.random.choice(['Мужской', 'Женский'], n_transactions),
    'Категория_товара': np.random.choice(['Электроника', 'Мода', 'Продукты', 'Дом', 'Спорт'], n_transactions),
    'Цена_за_единицу': np.random.uniform(10, 100, n_transactions).round(2),
    'Количество': np.random.randint(1, 10, n_transactions),
    'Дата': [datetime.date(2023, np.random.randint(1, 13), np.random.randint(1, 29)) for _ in range(n_transactions)]
}

# Создаем DataFrame
df = pd.DataFrame(data)

# Рассчитываем общую сумму
df['Итог'] = df['Цена_за_единицу'] * df['Количество']

# 2. Очистка данных
# Проверка пропущенных значений (в синтетических данных их нет, но для демонстрации)
print("Пропущенные значения:\n", df.isnull().sum())

# Преобразование столбца Дата в datetime
df['Дата'] = pd.to_datetime(df['Дата'])

# Проверка типов данных
print("\nТипы данных:\n", df.dtypes)

# 3. Анализ данных
# Средняя выручка с использованием numpy
avg_sales = np.mean(df['Итог'])
std_sales = np.std(df['Итог'])
print(f"\nСредняя выручка за транзакцию: {avg_sales:.2f}")
print(f"Стандартное отклонение выручки: {std_sales:.2f}")

# Группировка по категориям товаров
sales_by_category = df.groupby('Категория_товара')['Итог'].sum().sort_values(ascending=False)
print("\nВыручка по категориям:\n", sales_by_category)

# Группировка по городам
sales_by_city = df.groupby('Город')['Итог'].sum()
print("\nВыручка по городам:\n", sales_by_city)

# Корреляционная матрица для числовых столбцов
numeric_cols = ['Цена_за_единицу', 'Количество', 'Итог']
corr_matrix = np.corrcoef(df[numeric_cols].values.T)
print("\nКорреляционная матрица:\n", corr_matrix)

# 4. Визуализация
# График 1: Барплот выручки по категориям
plt.figure(figsize=(10, 6))
sns.barplot(x=sales_by_category.values, y=sales_by_category.index)
plt.title('Общая выручка по категориям товаров')
plt.xlabel('Выручка (₽)')
plt.ylabel('Категория товара')
plt.tight_layout()
plt.savefig('sales_by_category.png')
plt.show()

# График 2: Распределение выручки по городам
plt.figure(figsize=(8, 5))
sns.barplot(x=sales_by_city.index, y=sales_by_city.values)
plt.title('Выручка по городам')
plt.xlabel('Город')
plt.ylabel('Выручка (₽)')
plt.tight_layout()
plt.savefig('sales_by_city.png')
plt.show()

# График 3: Тепловая карта корреляций
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, xticklabels=numeric_cols, yticklabels=numeric_cols, cmap='coolwarm')
plt.title('Корреляционная матрица')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.show()

# График 4: Распределение общей выручки
plt.figure(figsize=(10, 6))
sns.histplot(df['Итог'], bins=30, kde=True)
plt.title('Распределение общей выручки')
plt.xlabel('Выручка (₽)')
plt.ylabel('Частота')
plt.tight_layout()
plt.savefig('sales_distribution.png')
plt.show()

# 5. Выводы
print("\nОсновные выводы:")
print(f"1. Самая прибыльная категория: {sales_by_category.index[0]} с выручкой {sales_by_category.values[0]:.2f} ₽")
print(f"2. Средняя выручка за транзакцию: {avg_sales:.2f} ₽")
print(f"3. Корреляция между ценой и количеством: {corr_matrix[0,1]:.2f}")
