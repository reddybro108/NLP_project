from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import os

router = APIRouter()

# Load model and vectorizer if available
MODEL_PATH = "sentiment_model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
vectorizer = joblib.load(VECTORIZER_PATH) if os.path.exists(VECTORIZER_PATH) else None

class TextIn(BaseModel):
    text: str

@router.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application!"}

@router.post("/predict")
async def predict_sentiment(data: TextIn):
    if not model or not vectorizer:
        return {"error": "Model or vectorizer not found. Train the model first."}
    text_vec = vectorizer.transform([data.text])
    prediction = model.predict(text_vec)[0]
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = sentiment_map.get(int(prediction), "Unknown")
    proba = model.predict_proba(text_vec)[0]
    score = float(max(proba))
    return {"sentiment": sentiment, "score": score}

@router.get("/health")
async def health_check():
    return {"status": "ok"}

@router.post("/batch_predict")
async def batch_predict(texts: list[TextIn]):
    results = []
    for item in texts:
        text_vec = vectorizer.transform([item.text])
        prediction = model.predict(text_vec)[0]
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = sentiment_map.get(int(prediction), "Unknown")
        proba = model.predict_proba(text_vec)[0]
        score = float(max(proba))
        results.append({"text": item.text, "sentiment": sentiment, "score": score})
    return results
