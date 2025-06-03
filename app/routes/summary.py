from fastapi import APIRouter, HTTPException
from app.models.summary import SummaryRequest, SummaryOut
import app.graphiti_client as graphiti_client
from loguru import logger

router = APIRouter()

@router.post("/conversation_summary", response_model=SummaryOut)
async def conversation_summary(payload: SummaryRequest):
    """
    Summarize the last `num_conversations` for a given user.
    """
    try:
        # Use Graphiti core summarization if available
        if getattr(graphiti_client, '_USE_GRAPHITI', False) and graphiti_client.graphiti:
            # Assume Graphiti client has a method `summarize_episodes`
            summary = await graphiti_client.graphiti.summarize_episodes(
                uid=payload.uid,
                num_conversations=payload.num_conversations
            )
        else:
            # Fallback: Graphiti core unavailable
            summary = f"Summary not available: Graphiti core is disabled."
        return SummaryOut(summary=summary)
    except Exception as e:
        logger.error(f"Error in conversation_summary for uid={payload.uid}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 