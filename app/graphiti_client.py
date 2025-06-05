import os
from dotenv import load_dotenv
import httpx
import json
import re
from neo4j import GraphDatabase
from loguru import logger
# Attempt to import Graphiti client and LLM client, with stubs if unavailable
try:
    from graphiti_core import Graphiti
    from graphiti_core.llm_client import OpenAIGenericClient
    _USE_GRAPHITI = True
except ImportError:
    class OpenAIGenericClient:
        def __init__(self, *args, **kwargs):
            pass

    class Graphiti:
        def __init__(self, *args, **kwargs):
            pass

        def add_episode(self, uid, conversation):
            # stub that returns dummy episode with id equal to uid
            class Episode:
                id = uid
            return Episode()
    _USE_GRAPHITI = False

# Environment variables
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("NEBIUS_MODEL_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
GRAPHITI_LLM_TIMEOUT = int(os.getenv("GRAPHITI_LLM_TIMEOUT", "25"))

# Honor USE_GRAPHITI override: set to "false" to disable Graphiti core ingestion and use manual fallback
USE_GRAPHITI_ENV = os.getenv("USE_GRAPHITI", "true").lower() in ("true","1","yes")
if not USE_GRAPHITI_ENV:
    logger.info("USE_GRAPHITI environment flag is false; disabling Graphiti core ingestion.")
    _USE_GRAPHITI = False

# Initialize LLM client
llm_client = OpenAIGenericClient(
    base_url=OPENAI_API_BASE,
    api_key=OPENAI_API_KEY,
    model_name=MODEL_NAME,
    timeout=GRAPHITI_LLM_TIMEOUT,
)

# Initialize Neo4j driver
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
)

# Initialize Graphiti client if available
if _USE_GRAPHITI:
    graphiti = Graphiti(driver=driver, llm_client=llm_client, embedding_model_name=EMBEDDING_MODEL_NAME)
else:
    graphiti = None

# In-memory store of conversations per user for fallback summarization
conversation_store: dict[str, list[list[dict]]] = {}

def add_episode(uid: str, conv: list[dict]) -> str:
    """
    Ingests a conversation as a Graphiti Episode and extracts multiple relationships.

    Returns the episode id.
    """
    logger.info(f"add_episode called with uid={uid}, num_turns={len(conv)}, USE_GRAPHITI={_USE_GRAPHITI}")
    if not _USE_GRAPHITI:
        logger.info(f"Using fallback manual ingestion for uid={uid}")
        with driver.session() as session:
            # Log raw user texts
            logger.debug(f"User turns: {[t.get('text','') for t in conv if t.get('speaker')=='User']}")
            # Ensure user node exists
            session.run("MERGE (u:User {uid:$uid})", uid=uid)
            # Prepare full conversation with speaker labels for LLM
            conv_formatted = "\n".join([f"{turn.get('speaker')}: {turn.get('text','')}" for turn in conv])
            logger.debug(f"conv_formatted for LLM: {conv_formatted}")
            # Ask LLM to extract relationships
            # Validate LLM endpoint and credentials
            if not OPENAI_API_BASE or not OPENAI_API_KEY:
                logger.error(f"Missing OPENAI_API_BASE or OPENAI_API_KEY; skipping LLM request for uid={uid}")
                return uid
            logger.info(f"Sending LLM request to {OPENAI_API_BASE}/chat/completions")
            system_instruction = (
                                "You are a relationship‑extraction assistant. Given the full conversation between "
                "'AI' and 'User', extract the user's expressed facts and preferences across **many** "
                "high‑level dimensions:\n"
                "  • problem           – the current issue, challenge, or pain point the user mentions\n"
                "  • religion          – stated or implied spiritual / religious beliefs or affiliations\n"
                "  • culture           – cultural background, traditions, or values the user identifies with\n"
                "  • tone              – the overall emotional tone the user conveys (e.g. anxious, optimistic)\n"
                "  • tone_sensitivity  – how sensitive the user appears to that tone (e.g. highly sensitive, mildly aware)\n"
                "  • theme_resonance   – topics, metaphors, or themes that visibly resonate with the user\n"
                "  • preference        – any explicit preference, coping strategy, or desire\n"
                "\n"
                "You can also extract other facts and preferences that are not listed here, but only if they are explicitly mentioned in the conversation."
                "For **each** distinct fact you find, append the  JSON object with these keys:\n"
                "  \"relation\"    – one of [\"PROBLEM\", \"RELIGION\", \"CULTURE\", \"TONE\", "
                "\"TONE_SENSITIVITY\", \"THEME_RESONANCE\", \"PREFERENCE\"]\n"
                "  \"object\"      – the extracted text snippet or a concise phrase capturing the fact\n"
                "  \"object_type\" – a PascalCase node label to create in Neo4j (use Problem, Religion, "
                "Culture, Tone, ToneSensitivity, ThemeResonance, Preference respectively)\n"
                "\n"
                "Return **one JSON array** (no additional text). Aim for 2‑3 objects per conversation turn, "
                "but never repeat identical facts. Adhere strictly to only one JSON array."
            )
            chat_payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": conv_formatted},
                ],
                "max_tokens": 500,
                "temperature": 0,
            }
            try:
                resp = httpx.post(
                    f"{OPENAI_API_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json=chat_payload,
                    timeout=GRAPHITI_LLM_TIMEOUT,
                )
            except Exception as e:
                logger.error(f"LLM request failed for uid={uid}: {e}")
                return uid
            resp.raise_for_status()
            logger.debug(f"LLM response status: {resp.status_code}")
            content = resp.json()["choices"][0]["message"]["content"]
            logger.debug(f"LLM response content: {content}")
            # Extract JSON array from LLM output
            start = content.find('[')
            end = content.rfind(']')
            if start != -1 and end != -1 and end > start:
                json_str = content[start:end+1]
                logger.debug(f"Extracted JSON string for parsing: {json_str}")
                try:
                    rels = json.loads(json_str)
                except Exception as e:
                    logger.error(f"JSON parse error: {e}; json_str: {json_str}")
                    rels = []
            else:
                logger.error(f"No JSON array found in LLM output: {content}")
                rels = []
            logger.debug(f"Extracted relationships: {rels}")
            # Create nodes and relationships based on LLM output
            rel_count = 0
            for rel in rels:
                # Determine node label and object text
                raw_obj_type = rel.get("object_type", "Preference")
                obj_type = raw_obj_type.strip().replace(' ', '_').capitalize()
                obj = rel.get("object", "")
                # Determine relationship type, mapping preferences to LIKES
                if obj_type == "Preference":
                    rel_type = "LIKES"
                else:
                    raw_rel = rel.get("relation", "REL")
                    rel_type = re.sub(r"\W+", "_", raw_rel).upper()
                logger.info(f"Creating relationship {uid}-[:{rel_type}]->{obj_type}({obj})")
                # Merge object node
                session.run(f"MERGE (o:{obj_type} {{name:$obj}})", obj=obj)
                # Link user to object with sanitized rel_type
                session.run(
                    f"MATCH (u:User {{uid:$uid}}), (o:{obj_type} {{name:$obj}}) MERGE (u)-[:{rel_type}]->(o)",
                    uid=uid, obj=obj,
                )
                rel_count += 1
            logger.info(f"Total relationships created for uid={uid}: {rel_count}")
            # Store conversation for summarization fallback
            conversation_store.setdefault(uid, []).append(conv)
        return uid
    # Use Graphiti to ingest conversation and extract relationships into Neo4j
    logger.info(f"Using Graphiti ingestion for uid={uid}")
    try:
        episode = graphiti.add_episode(uid=uid, conversation=conv)
    except Exception as e:
        logger.error(f"Graphiti.add_episode failed for uid={uid}: {e}")
        raise
    logger.info(f"Graphiti.add_episode returned episode_id={episode.id}")
    return episode.id

# Add LLM-based question generation and Graphiti preference search
SYSTEM_PROMPT = "You are a helpful assistant that generates the next follow-up question to learn about user preferences. Only output the question."

async def generate_next_question(preferences: list[str]) -> str:
    """
    Generate the next dynamic question given existing preferences using LLM,
    trying chat/completions first, then falling back to completions endpoint.
    """
    pref_text = ", ".join(preferences) if preferences else ""
    if pref_text:
        user_prompt = f"User preferences so far: {pref_text}. Ask the next question to learn another preference."
    else:
        user_prompt = "Ask a question to learn about the user's preferences."
    async with httpx.AsyncClient(timeout=GRAPHITI_LLM_TIMEOUT) as client:
        # First, try chat/completions
        chat_payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            resp = await client.post(
                f"{OPENAI_API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json=chat_payload,
            )
            resp.raise_for_status()
            chat_data = resp.json()
            return chat_data["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as chat_err:
            if chat_err.response.status_code != 404:
                raise
        # Fallback to legacy completions
        comp_payload = {
            "model": MODEL_NAME,
            "prompt": f"{SYSTEM_PROMPT}\n\n{user_prompt}",
            "max_tokens": 64,
        }
        resp2 = await client.post(
            f"{OPENAI_API_BASE}/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json=comp_payload,
        )
        resp2.raise_for_status()
        comp_data = resp2.json()
        return comp_data.get("choices", [{}])[0].get("text", "").strip()

def get_preferences(uid: str, top_k: int = 5) -> list[str]:
    """
    Retrieve top_k preferences for a user using Graphiti hybrid search recipe.
    """
    with driver.session() as session:
        # Return preference node names via LIKES relationship
        result = session.run(
            "MATCH (u:User {uid:$uid})-[:LIKES]->(p:Preference) RETURN p.name AS text LIMIT $k",
            uid=uid, k=top_k,
        )
        return [record["text"] for record in result]

async def summarize_conversation(uid: str, conv: list[dict]) -> str:
    """
    Summarize the given conversation using LLM.
    """
    # Format conversation turns
    conv_formatted = "\n".join([f"{turn.get('speaker')}: {turn.get('text','')}" for turn in conv])
    system_prompt = (
        "You are a helpful assistant that summarizes the following conversation between AI and User concisely. "
        "Only output the summary."
    )
    async with httpx.AsyncClient(timeout=GRAPHITI_LLM_TIMEOUT) as client:
        chat_payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conv_formatted},
            ],
            "max_tokens": 256,
        }
        try:
            resp = await client.post(
                f"{OPENAI_API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json=chat_payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as err:
            if err.response.status_code != 404:
                raise
        # Fallback to legacy completions endpoint
        comp_payload = {
            "model": MODEL_NAME,
            "prompt": f"{system_prompt}\n\n{conv_formatted}",
            "max_tokens": 256,
        }
        resp2 = await client.post(
            f"{OPENAI_API_BASE}/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json=comp_payload,
        )
        resp2.raise_for_status()
        comp_data = resp2.json()
        return comp_data.get("choices", [{}])[0].get("text", "").strip() 

# --------------- Additional commented-out preference retrieval strategies ---------------
# def get_preferences_by_recent_conversations(uid: str, num_conversations: int = 2) -> list[str]:
#     """
#     Retrieve preferences from the last `num_conversations` episodes for a user.
#     """
#     with driver.session() as session:
#         # Assume Episode nodes have a `created_at` property
#         result = session.run(
#             """
#             MATCH (u:User {uid:$uid})-[:CREATED]->(e:Episode)
#             WITH e ORDER BY e.created_at DESC LIMIT $n
#             MATCH (e)-[]->(p:Preference)
#             RETURN p.text AS text
#             """,
#             uid=uid, n=num_conversations
#         )
#         return [record["text"] for record in result]
#
# def get_preferences_with_context(uid: str, previous_question: str, top_k: int = 5) -> list[str]:
#     """
#     Retrieve top_k preferences using vector search based on previous question context.
#     """
#     # Example using Graphiti hybrid search recipe:
#     # prefs = graphiti.search_preferences(uid=uid, context=previous_question, top_k=top_k)
#     # return prefs
#     pass 