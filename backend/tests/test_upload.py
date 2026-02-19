import pandas as pd
from utils.validators import validate_csv

def test_csv_validation():
    df = pd.DataFrame({
        "income":[30000],
        "loan":[20000],
        "emi":[3000],
        "tenure":[6],
        "ontime":[4],
        "delays":[1],
        "credit":[700]
    })

    assert validate_csv(df) is True
