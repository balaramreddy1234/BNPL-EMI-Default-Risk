def check_bias(predictions):
    return {
        "total": len(predictions),
        "high_risk": sum(predictions)
    }
