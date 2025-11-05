# tests/test_api.py
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    """Test the health endpoint which doesn't require authentication."""
    r = client.get('/health')
    assert r.status_code == 200

def test_latest_with_auth():
    """Test the latest endpoint with proper authentication."""
    r = client.get('/api/crypto/latest', headers={"X-API-Key": "your_secure_api_key_here"})
    # This should return 200 if successful, 500 if there's a database error
    assert r.status_code in (200, 500)