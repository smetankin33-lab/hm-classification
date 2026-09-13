import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.predictor import predict, load_model

client = TestClient(app)

# Тест 1: Проверка самой функции предсказания (без HTTP)
def test_predict_function():
    load_model()
    result = predict("Fitted top", "Cotton jersey with long sleeves")
    assert "predicted_group" in result
    assert "probabilities" in result
    assert isinstance(result["probabilities"], dict)
    assert len(result["probabilities"]) == 5

# Тест 2: Эндпоинт /health
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# Тест 3: Эндпоинт /predict (успешный запрос)
def test_predict_success():
    payload = {
        "prod_name": "Fitted top",
        "detail_desc": "Fitted top in stretch organic cotton jersey with a polo neck and long sleeves."
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_group" in data
    assert "probabilities" in data

# Тест 4: Некорректный ввод (пустой JSON)
def test_predict_invalid_payload():
    response = client.post("/predict", json={})
    # Pydantic использует default="", поэтому статус будет 200, а не 422.
    # Главное, что сервис не падает с 500 ошибкой.
    assert response.status_code in [200, 422]

# Тест 5: Пустой текст (не должен приводить к аварийному завершению)
def test_predict_empty_text():
    payload = {
        "prod_name": "",
        "detail_desc": ""
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_group" in data

# Тест 6: Параметризованный тест
@pytest.mark.parametrize("name, desc, expected_status", [
    ("Dress", "Short fitted dress", 200),
    ("Sweater", "Knitted wool", 200),
    ("", "Only description here", 200)
])
def test_predict_parametrized(name, desc, expected_status):
    payload = {"prod_name": name, "detail_desc": desc}
    response = client.post("/predict", json=payload)
    assert response.status_code == expected_status