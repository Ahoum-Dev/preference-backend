import pytest
from fastapi.testclient import TestClient
from app.main import app
import app.graphiti_client as gc

# Mock out dependencies for next_question
@pytest.fixture(autouse=True)
def mock_graphiti_calls(monkeypatch):
    # Mock get_preferences to return a fixed list
    monkeypatch.setattr(gc, "get_preferences", lambda uid, top_k: ["pref1", "pref2", "pref3"])
    # Mock generate_next_question to return a predictable question
    async def dummy_generate(preferences):
        return "dummy question"
    monkeypatch.setattr(gc, "generate_next_question", dummy_generate)

client = TestClient(app)

def test_next_question():
    payload = {"uid": "user123", "num_preferences": 3}
    response = client.post("/next_question", json=payload)
    assert response.status_code == 200
    assert response.json() == {"question": "dummy question"}

@pytest.mark.skip(reason="Context-aware route not implemented yet")
def test_next_question_with_context():
    # This test will validate /next_question_with_context when enabled
    payload = {
        "uid": "user123",
        "previous_question": "What calms you?",
        "num_preferences": 2
    }
    response = client.post("/next_question_with_context", json=payload)
    assert response.status_code == 200
    assert response.json() == {"question": "dummy question"} 