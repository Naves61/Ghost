from fastapi.testclient import TestClient
from app.api import app, INT


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


def test_soc_cadence_accepts_float():
    client = TestClient(app)
    resp = client.post("/config/soc_cadence?seconds=0.3")
    assert resp.status_code == 200
    assert resp.json()["soc_cadence"] == 0.3


def test_answer_interrupt_runs_soc():
    client = TestClient(app)
    qid = INT.create("Need data", "r", [])
    resp = client.post(f"/interrupts/answer?qid={qid}&text=42")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["wm"][-1]["content"] == data["thought"]["content"]