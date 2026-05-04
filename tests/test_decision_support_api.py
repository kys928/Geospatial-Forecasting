from fastapi.testclient import TestClient

from plume.api.main import create_app


def test_decision_support_latest_no_session():
    app = create_app()
    client = TestClient(app)
    resp = client.get('/decision-support/latest')
    assert resp.status_code == 200
    body = resp.json()
    assert body['mode'] == 'stub'
    assert 'No active session' in body['briefing']


def test_decision_support_chat_grounded_stub():
    app = create_app()
    client = TestClient(app)
    resp = client.post('/decision-support/chat', json={'message': 'What should we do?'})
    assert resp.status_code == 200
    body = resp.json()
    assert 'Grounded response' in body['answer']
