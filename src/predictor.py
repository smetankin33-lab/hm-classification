import joblib
import os

# Путь к модели (относительно корня проекта)
# Мы используем os.path, чтобы путь корректно определялся независимо от того, откуда запущен скрипт
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lr_pipeline.joblib')

model = None


def load_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
    return model


def predict(name: str, description: str) -> dict:
    load_model()

    # Обработка пустых строк (требование ТЗ: пустой текст не должен приводить к падению)
    name = name if name else ""
    description = description if description else ""

    # Объединяем тексты так же, как мы делали это на Шаге 2 при обучении
    text = (name + " " + description).strip()

    # Если после очистки текст абсолютно пустой, передаем пробел, чтобы TF-IDF не упал с ошибкой
    if not text:
        text = " "

        # Предсказание класса
    pred_class = model.predict([text])[0]

    # Получение вероятностей
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_

    # Формируем словарь вероятностей
    prob_dict = {str(c): float(p) for c, p in zip(classes, probabilities)}

    return {
        "predicted_group": str(pred_class),
        "probabilities": prob_dict
    }