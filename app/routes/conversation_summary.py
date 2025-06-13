from fastapi import APIRouter, HTTPException, Query
from app.models.conversation import ConversationIn
import app.graphiti_client as graphiti_client

router = APIRouter()

@router.post("/conversation_summary")
async def conversation_summary(payload: ConversationIn):
    """
    Generates a concise summary of the given conversation using the LLM.
    """
    try:
        # Convert conversation turns to list of dicts
        conv_list = [turn.dict() for turn in payload.conversation]
        summary = await graphiti_client.summarize_conversation(uid=payload.uid, conv=conv_list)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

