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
    proba = model.predict_proba(text_vec)[0]
    score = float(proba[1]) if len(proba) > 1 else float(proba[0])
    threshold = 0.2
    if len(proba) == 2:
        # Binary model with neutral threshold
        if abs(score - 0.5) <= threshold:
            sentiment = "Neutral"
        elif score > 0.5:
            sentiment = "Positive"
        else:
            sentiment = "Negative"
    else:
        # Multi-class model
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = sentiment_map.get(int(model.predict(text_vec)[0]), "Unknown")
    return {"sentiment": sentiment, "score": score}
