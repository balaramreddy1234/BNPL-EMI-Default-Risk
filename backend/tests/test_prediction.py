from ml.predict import predict_single

def test_predict():
    sample = {
        "income": 30000,
        "loan": 20000,
        "emi": 3500,
        "tenure": 8,
        "ontime": 3,
        "delays": 4,
        "credit": 610,
        "emi_income_ratio": 0.11,
        "delay_ratio": 0.5
    }

    result = predict_single(sample)
    assert result in ["HIGH", "LOW"]
