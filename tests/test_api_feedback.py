from fastapi.testclient import TestClient
from app.api import app


def test_ingest_returns_thought_and_wm():
    client = TestClient(app)
    stimulus = {
        "id": "stim-test",
        "source": "unit-test",
        "content": "Study linear algebra",
        "metadata": {},
        "ts": 0,
    }
    resp = client.post("/stimuli", json=[stimulus])
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    thought = data.get("thought", {})
    assert "content" in thought
    # thought references the stimulus
    assert "linear algebra" in thought["content"].lower()
    # working memory last entry matches thought content
    assert data["wm"][-1]["content"] == thought["content"]
    assert isinstance(data.get("stored"), bool)
