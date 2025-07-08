import os
import joblib
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Load model and vectorizer if available
MODEL_PATH = "sentiment_model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
vectorizer = joblib.load(VECTORIZER_PATH) if os.path.exists(VECTORIZER_PATH) else None


class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    sentiment: str
    score: float


class BatchPredictionIn(BaseModel):
    texts: List[TextIn]


class BatchPredictionOut(BaseModel):
    text: str
    sentiment: str
    score: float


@router.get("/", summary="Root endpoint")
async def read_root():
    return {"message": "Welcome to the FastAPI application!"}


@router.get("/items/{item_id}", summary="Get item with optional query")
async def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "query": q}


@router.post("/predict", response_model=PredictionOut, summary="Predict sentiment for a single text")
async def predict_sentiment(data: TextIn):
    if not model or not vectorizer:
        logger.error("Model or vectorizer not found.")
        raise HTTPException(
            status_code=503,
            detail="Model or vectorizer not found. Train the model first."
        )

    text_vec = vectorizer.transform([data.text])
    prediction = model.predict(text_vec)[0]
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = sentiment_map.get(int(prediction), "Unknown")
    proba = model.predict_proba(text_vec)[0]
    score = float(max(proba))

    return {"sentiment": sentiment, "score": score}


@router.get("/health", summary="Health check")
async def health_check():
    return {"status": "ok"}


@router.post("/batch_predict", response_model=List[BatchPredictionOut], summary="Predict sentiment for a batch of texts")
async def batch_predict(batch_data: BatchPredictionIn):
    if not model or not vectorizer:
        logger.error("Model or vectorizer not found for batch prediction.")
        raise HTTPException(
            status_code=503,
            detail="Model or vectorizer not found. Train the model first."
        )

    results = []
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    for item in batch_data.texts:
        text_vec = vectorizer.transform([item.text])
        prediction = model.predict(text_vec)[0]
        sentiment = sentiment_map.get(int(prediction), "Unknown")
        proba = model.predict_proba(text_vec)[0]
        score = float(max(proba))
        results.append({
            "text": item.text,
            "sentiment": sentiment,
            "score": score
        })

    return results
