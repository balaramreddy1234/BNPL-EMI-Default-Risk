from app import app

def test_predict_api():
    client = app.test_client()

    response = client.post("/predict", json={
        "income": 40000,
        "loan": 15000,
        "emi": 2500,
        "tenure": 6,
        "ontime": 5,
        "delays": 1,
        "credit": 720
    })

    assert response.status_code == 200
