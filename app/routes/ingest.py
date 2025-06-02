from fastapi import APIRouter, HTTPException, status
from app.models.conversation import ConversationIn
import app.graphiti_client as graphiti_client

router = APIRouter()

@router.post("/ingest_conversation", status_code=status.HTTP_201_CREATED)
async def ingest_conversation(payload: ConversationIn):
    """
    Ingest a seeker-AI conversation and store as a Graphiti episode.
    """
    try:
        conv_list = [{"speaker": turn.speaker, "text": turn.text} for turn in payload.conversation]
        episode_id = graphiti_client.add_episode(uid=payload.uid, conv=conv_list)
        return {"status": "ok", "episode_id": episode_id}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) 