from fastapi import FastAPI
from app.api.routes import router as api_router

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application!"}

app.include_router(api_router)