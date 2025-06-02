from fastapi import APIRouter, HTTPException
from app.models.question import QuestionOut
from app.models.question_request import NextQuestionIn
import app.graphiti_client as graphiti_client

router = APIRouter()

@router.post("/next_question", response_model=QuestionOut)
async def next_question(payload: NextQuestionIn):
    """
    Generates the next dynamic question based on user's long-term preferences.
    """
    try:
        prefs = graphiti_client.get_preferences(uid=payload.uid, top_k=payload.num_preferences)
        question_text = await graphiti_client.generate_next_question(preferences=prefs)
        return QuestionOut(question=question_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 