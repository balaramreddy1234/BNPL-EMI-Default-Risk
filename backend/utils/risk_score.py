def classify_risk(probability):
    """
    Categorizes risk based on probability thresholds:
    - Low: 0% to 30%
    - Medium: 30% to 70%
    - High: 70% to 100%
    """
    if probability >= 0.70:
        return "HIGH RISK"
    elif probability >= 0.30:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"