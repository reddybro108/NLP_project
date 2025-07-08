from fastapi import APIRouter
import joblib
import os
from predict import TextIn, BatchTextIn

router = APIRouter()

MODEL_PATH = "sentiment_model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
vectorizer = joblib.load(VECTORIZER_PATH) if os.path.exists(VECTORIZER_PATH) else None

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}


@router.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application!"}


@router.post("/predict")
async def predict_sentiment(data: TextIn):
    if not model or not vectorizer:
        return {"error": "Model or vectorizer not found. Train the model first."}

    text_vec = vectorizer.transform([data.text])
    prediction = model.predict(text_vec)[0]
    sentiment = label_map.get(prediction, "Unknown")
    score = float(max(model.predict_proba(text_vec)[0]))

    return {"sentiment": sentiment, "score": score}


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.post("/batch_predict")
async def batch_predict(texts: BatchTextIn):
    results = []

    for item in texts.texts:
        text_vec = vectorizer.transform([item.text])
        prediction = model.predict(text_vec)[0]
        sentiment = label_map.get(prediction, "Unknown")
        score = float(max(model.predict_proba(text_vec)[0]))
        results.append({"text": item.text, "sentiment": sentiment, "score": score})

    return results
