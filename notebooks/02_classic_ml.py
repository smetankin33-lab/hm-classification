import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import os

# Загрузка очищенных данных
df = pd.read_csv('../data/articles_cleaned.csv')
print(f" Загружено строк: {df.shape[0]}")

# 1. Объединение текстовых признаков
# Заполняем возможные пропуски пустыми строками перед конкатенацией
df['text'] = df['prod_name'].fillna('') + ' ' + df['detail_desc'].fillna('')

X = df['text']
y = df['garment_group_name']

# 2. Разделение на train и test (фиксированный random_state, как в ТЗ)
# stratify=y сохраняет пропорции классов в train и test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ==========================================
# 3. Базовая модель DummyClassifier
# ==========================================
dummy_clf = DummyClassifier(strategy='most_frequent', random_state=42)
dummy_clf.fit(X_train, y_train)
y_pred_dummy = dummy_clf.predict(X_test)

dummy_acc = accuracy_score(y_test, y_pred_dummy)
dummy_f1 = f1_score(y_test, y_pred_dummy, average='macro')
print(f"\n--- DummyClassifier ---")
print(f"Accuracy: {dummy_acc:.4f}")
print(f"Macro-F1: {dummy_f1:.4f}")

# ==========================================
# 4. Классическая модель TF-IDF + LogisticRegression
# ==========================================
# Используем Pipeline. Это критично для ТЗ: сервис сможет выполнять предсказание без повторного обучения,
# так как Pipeline хранит в себе и TF-IDF векторайзер, и саму модель.
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, stop_words='english', ngram_range=(1, 2))),
    ('clf', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
])

print("\n Обучение TF-IDF + LogisticRegression (может занять пару секунд)...")
pipeline.fit(X_train, y_train)
y_pred_lr = pipeline.predict(X_test)

lr_acc = accuracy_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr, average='macro')
print(f"\n--- LogisticRegression ---")
print(f"Accuracy: {lr_acc:.4f}")
print(f"Macro-F1: {lr_f1:.4f}")

# ==========================================
# 5. Матрица ошибок
# ==========================================
cm = confusion_matrix(y_test, y_pred_lr, labels=pipeline.classes_)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок (Logistic Regression)')
plt.tight_layout()
plt.show()

# ==========================================
# 6. Анализ ошибок (Минимум 3 ошибочных предсказания по ТЗ)
# ==========================================
errors_mask = y_test != y_pred_lr
errors_df = pd.DataFrame({
    'text': X_test[errors_mask],
    'true_class': y_test[errors_mask],
    'pred_class': y_pred_lr[errors_mask]
})

print(f"\n Всего ошибок на тесте: {len(errors_df)}")
print("\n--- Примеры 3 ошибочных предсказаний ---")
# Обрезаем текст для удобного вывода в консоль
errors_to_show = errors_df.head(3).copy()
errors_to_show['text'] = errors_to_show['text'].apply(lambda x: x[:100] + '...')
print(errors_to_show.to_string())

# ==========================================
# СОХРАНЕНИЕ МОДЕЛИ
# ==========================================
os.makedirs('../models', exist_ok=True)
joblib.dump(pipeline, '../models/lr_pipeline.joblib')
print("\n Модель LogisticRegression сохранена в models/lr_pipeline.joblib")