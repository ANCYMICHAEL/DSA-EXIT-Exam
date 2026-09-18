import pytest

from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_home_page_loads(client):
    resp = client.get('/')
    assert resp.status_code == 200
    assert b'Pokemon Type Predictor' in resp.data


def test_predict_route_works(client):
    payload = {
        'height_dm': 5,
        'weight_hg': 80,
        'base_experience': 64,
        'hp': 45,
        'attack': 49,
        'defense': 49,
    }
    resp = client.post('/predict', data=payload)
    assert resp.status_code == 200
    assert b'Predicted Type' in resp.data
