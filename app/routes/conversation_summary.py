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

# New endpoint: get last n full conversations for a user
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