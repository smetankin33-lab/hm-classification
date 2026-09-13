# Используем официальный легковесный образ Python
FROM python:3.11-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# ШАГ 1: Копируем ТОЛЬКО requirements.txt

COPY requirements.txt .

# ШАГ 2: Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# ШАГ 3: Копируем исходный код
COPY src/ ./src/

# ШАГ 4: Копируем ТОЛЬКО нужную обученную модель

COPY models/lr_pipeline.joblib ./models/lr_pipeline.joblib

# Указываем порт, который слушает приложение
EXPOSE 8000

# Команда запуска контейнера
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]