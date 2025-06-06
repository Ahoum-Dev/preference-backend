from fastapi import APIRouter, HTTPException
from app.models.summary import SummaryRequest, SummaryOut
import app.graphiti_client as graphiti_client
import httpx
import json
from loguru import logger

router = APIRouter()

@router.post("/conversation_content", response_model=SummaryOut)
async def conversation_content(payload: SummaryRequest):
    """
    Produce a content‐style summary (using a different system prompt) for the last `num_conversations` for a given user.
    """
    try:
        # Pull raw conversations from Neo4j
        with graphiti_client.driver.session() as session:
            result = session.run(
                "MATCH (u:User {uid:$uid})-[:CREATED]->(e:Episode) "
                "RETURN e.conversation AS conv_json ORDER BY e.created_at DESC LIMIT $n",
                uid=payload.uid, n=payload.num_conversations
            )
            convs = [json.loads(record["conv_json"]) for record in result]
        if not convs:
            summary = f"No conversations found for user {payload.uid}."
        else:
            # Flatten to text
            text_blocks = []
            for conv in convs:
                lines = [f"{turn.get('speaker')}: {turn.get('text','')}" for turn in conv]
                text_blocks.append("\n".join(lines))
            convo_text = "\n\n".join(text_blocks)
            # Build LLM prompt with a different system message
            prompt = f"Create a rich, detailed content summary of these user–AI interactions:\n\n{convo_text}"
            async with httpx.AsyncClient(timeout=graphiti_client.GRAPHITI_LLM_TIMEOUT) as client:
                resp = await client.post(
                    f"{graphiti_client.OPENAI_API_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {graphiti_client.OPENAI_API_KEY}"},
                    json={
                        "model": graphiti_client.MODEL_NAME,
                        "messages": [
                            {"role": "system", "content": "You are a content creation assistant. Produce an in-depth summary for content publication."},
                            {"role": "user", "content": prompt},
                        ],
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                summary = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return SummaryOut(summary=summary)
    except Exception as e:
        logger.error(f"Error in conversation_content for uid={payload.uid}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
