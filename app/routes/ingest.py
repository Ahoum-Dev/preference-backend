from fastapi import APIRouter, HTTPException, status
from app.models.conversation import ConversationIn
import app.graphiti_client as graphiti_client
from fastapi import APIRouter, HTTPException, Query

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
    
@router.get("/get_conversations")
def get_conversations(
    uid: str = Query(..., description="User ID"),
    n: int = Query(1, description="Number of most recent conversations to return")
):
    """
    Return the last n full conversations (as lists of turns) for a user.
    """
    with graphiti_client.driver.session() as session:
        result = session.run(
            """
            MATCH (u:User {uid:$uid})-[:CREATED]->(e:Episode)
            RETURN e.conversation AS conv_json, e.created_at AS created
            ORDER BY e.created_at DESC LIMIT $n
            """,
            uid=uid, n=n
        )
        conversations = []
        for record in result:
            conv_json = record["conv_json"]
            try:
                conv = graphiti_client.json.loads(conv_json)
            except Exception:
                conv = conv_json  # fallback: raw string
            conversations.append(conv)
        if not conversations:
            raise HTTPException(status_code=404, detail="No conversations found for user")
        return {"conversations": conversations} 
    