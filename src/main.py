from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predictor import predict, load_model

app = FastAPI(title="H&M Classification API", description="Сервис классификации товаров H&M")

# Pydantic модель для валидации входных данных
class ProductInput(BaseModel):
    prod_name: str = Field(default="", description="Название товара (prod_name)")
    detail_desc: str = Field(default="", description="Описание товара (detail_desc)")

# Pydantic модель для ответа
class PredictionResponse(BaseModel):
    predicted_group: str
    probabilities: dict

@app.on_event("startup")
async def startup_event():
    # Загружаем модель в память при старте приложения
    load_model()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_product(input_data: ProductInput):
    try:
        result = predict(input_data.prod_name, input_data.detail_desc)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")