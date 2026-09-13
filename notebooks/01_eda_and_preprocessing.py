import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split  # Импортируем функцию для стратифицированной выборки

# Настраиваем стиль графиков
sns.set_theme(style="whitegrid")

# Путь к данным
DATA_PATH = '../data/articles.csv'

print(" Загрузка данных...")
df = pd.read_csv(DATA_PATH)
print(f"Изначальный размер: {df.shape}")

# 1. Оставляем только нужные колонки
cols_to_keep = ['prod_name', 'detail_desc', 'garment_group_name']
df = df[cols_to_keep]

# 2. Удаляем строки с пропусками в этих колонках
df = df.dropna(subset=cols_to_keep)
print(f"Размер после удаления пропусков: {df.shape}")

# 3. Оставляем топ-5 самых частых групп
class_counts = df['garment_group_name'].value_counts()
top_5_classes = class_counts.head(5).index
df_filtered = df[df['garment_group_name'].isin(top_5_classes)]
print(f"Размер после фильтрации топ-5 классов: {df_filtered.shape}")

# 4. Ограничиваем до 10 000 товаров (с фиксированным random_state для воспроизводимости)
# ИСПРАВЛЕНИЕ: Используем train_test_split, так как pandas.sample не умеет стратифицировать
if len(df_filtered) > 10000:
    print("Выполняется стратифицированная выборка 10000 примеров...")
    # train_size=10000 означает, что мы берем первые 10000 примеров в выборку
    # stratify гарантирует сохранение пропорций классов
    df_sampled, _ = train_test_split(
        df_filtered,
        train_size=10000,
        random_state=42,
        stratify=df_filtered['garment_group_name']
    )
    df_filtered = df_sampled

print(f" Итоговый размер выборки: {df_filtered.shape}")
print(f"\n Распределение классов:\n{df_filtered['garment_group_name'].value_counts()}")

# ==========================================
# ВИЗУАЛИЗАЦИИ
# ==========================================

# Визуализация 1: Распределение классов (Barplot)
plt.figure(figsize=(10, 6))
sns.countplot(y='garment_group_name', data=df_filtered,
              order=df_filtered['garment_group_name'].value_counts().index,
              palette='viridis')
plt.title('Распределение товаров по группам (Топ-5 классов)', fontsize=14)
plt.xlabel('Количество товаров')
plt.ylabel('Группа одежды')
plt.tight_layout()
plt.show()

# Визуализация 2: Анализ длины описания (Histplot + KDE)
df_filtered['desc_length'] = df_filtered['detail_desc'].apply(len)

plt.figure(figsize=(12, 6))
sns.histplot(data=df_filtered, x='desc_length', hue='garment_group_name',
             bins=50, kde=True, element="step", stat="density", common_norm=False)
plt.title('Распределение длины описания товаров по классам', fontsize=14)
plt.xlabel('Длина описания (количество символов)')
plt.ylabel('Плотность')
plt.xlim(0, 1000) # Ограничим ось X для наглядности, если есть очень длинные тексты
plt.tight_layout()
plt.show()

# ==========================================
# СОХРАНЕНИЕ ПОДГОТОВЛЕННЫХ ДАННЫХ
# ==========================================
# Сохраним очищенный датасет, чтобы не гонять его каждый раз
os.makedirs('../data', exist_ok=True)
df_filtered.to_csv('../data/articles_cleaned.csv', index=False)
print(" Очищенные данные сохранены в data/articles_cleaned.csv")