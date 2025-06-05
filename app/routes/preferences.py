from fastapi import APIRouter, HTTPException
import app.graphiti_client as graphiti_client
import httpx
import json
from loguru import logger
from app.models.preferences import PreferencesRequest, PreferencesOut

router = APIRouter()

@router.post("/recent_preferences", response_model=PreferencesOut)
async def recent_preferences(payload: PreferencesRequest):
    """
    Extract user preferences from the last `num_conversations` for a given user.
    """
    try:
        # Use Graphiti core if available
        if getattr(graphiti_client, '_USE_GRAPHITI', False) and graphiti_client.graphiti:
            prefs = graphiti_client.get_preferences_by_recent_conversations(
                uid=payload.uid,
                num_conversations=payload.num_conversations
            )
            return PreferencesOut(preferences=prefs)
        # Fallback: use LLM on stored conversations
        convs = graphiti_client.conversation_store.get(payload.uid, [])
        if not convs:
            return PreferencesOut(preferences=[])
        last_convs = convs[-payload.num_conversations:]
        # Flatten conversations to text
        convo_text = "\n\n".join([
            "\n".join([f"{turn['speaker']}: {turn['text']}" for turn in conv])
            for conv in last_convs
        ])
        system_instruction = (
            "You are a preference extraction assistant. "
            "Given the following conversations between 'AI' and 'User', extract all distinct preferences the user expresses. "
            "Output a JSON array of strings."
        )
        chat_payload = {
            "model": graphiti_client.MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": convo_text},
            ],
            "temperature": 0,
        }
        async with httpx.AsyncClient(timeout=graphiti_client.GRAPHITI_LLM_TIMEOUT) as client:
            resp = await client.post(
                f"{graphiti_client.OPENAI_API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {graphiti_client.OPENAI_API_KEY}"},
                json=chat_payload,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        # Parse JSON response
        start = content.find('[')
        end = content.rfind(']')
        prefs = []
        if start != -1 and end != -1 and end > start:
            try:
                prefs = json.loads(content[start:end+1])
            except Exception as e:
                logger.error(f"JSON parse error in recent_preferences: {e}")
        return PreferencesOut(preferences=prefs)
    except Exception as e:
        logger.error(f"Error in recent_preferences for uid={payload.uid}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 