from fastapi import APIRouter

from app.schemas.search_schema import SearchRequest
from app.services.search_service import semantic_search
from app.services.answer_service import generate_answer

router = APIRouter()


@router.post("/search")
def search(request: SearchRequest):

    chunks = semantic_search(request.query)

    answer = generate_answer(request.query, chunks)

    return {
        "query": request.query,
        "answer": answer
    
    }