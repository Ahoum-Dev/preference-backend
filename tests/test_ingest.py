import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(autouse=True)
def mock_graphiti(monkeypatch):
    import app.graphiti_client as gc
    monkeypatch.setattr(gc, "add_episode", lambda uid, conv: "dummy_episode_id")

client = TestClient(app)

def test_ingest_conversation():
    payload = {
        "uid": "1234567890",
        "conversation": [
        {"speaker": "AI",    "text": "Hi there! What’s on your mind today?"},
        {"speaker": "User",  "text": "Honestly, I’ve been feeling anxious about my exams."},
        {"speaker": "AI",    "text": "That sounds stressful.  Could you tell me what part worries you most?"},
        {"speaker": "User",  "text": "I’m afraid of failing and letting my parents down."},
        {"speaker": "AI",    "text": "When you imagine that happening, what emotions come up?"},
        {"speaker": "User",  "text": "Mostly panic, and a bit of anger at myself."},
        {"speaker": "AI",    "text": "Have you found anything that calms you when panic rises?"},
        {"speaker": "User",  "text": "Deep breathing helps, but it doesn’t always last."},
        {"speaker": "AI",    "text": "Let’s note that coping tool.  Would you like to explore other strategies?"},
        {"speaker": "User",  "text": "Yes, I’d like that—maybe something that builds confidence."}
    
        ],
        "conversation_id": "1234567890",
        "created_at": "2025-06-02T12:00:00Z",
        "updated_at": "2025-06-02T12:00:00Z"
    }
    response = client.post("/ingest_conversation", json=payload)
    assert response.status_code == 201
    assert response.json() == {"status": "ok", "episode_id": "dummy_episode_id"} 