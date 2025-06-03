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

# --------------- Commented-out alternative route for context-aware question generation ---------------
# from app.models.question_request_with_context import NextQuestionWithContextIn
# 
# @router.post("/next_question_with_context", response_model=QuestionOut)
# async def next_question_with_context(payload: NextQuestionWithContextIn):
#     """
#     Generates the next dynamic question based on user's preferences and previous question context.
#     """
#     try:
#         # Use vector-search-based preference retrieval with previous question context
#         prefs = graphiti_client.get_preferences_with_context(
#             uid=payload.uid,
#             previous_question=payload.previous_question,
#             top_k=payload.num_preferences
#         )
#         question_text = await graphiti_client.generate_next_question(preferences=prefs)
#         return QuestionOut(question=question_text)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e)) 