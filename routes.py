from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def read_root():
<<<<<<< HEAD
    return {"message": "Welcome to the FastAPI application!"}
=======
    return {"message": "Welcome to the FastAPI application!"}

@router.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "query": q}
>>>>>>> 68af592e5555dfa1add2e0803f1b0b6a11d4b287
