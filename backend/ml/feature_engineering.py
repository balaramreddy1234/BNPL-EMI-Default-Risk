def engineer_features(df):
    df["emi_income_ratio"] = df["emi"] / (df["income"] + 1)
    df["delay_ratio"] = df["delays"] / (df["ontime"] + df["delays"] + 1)
    return df
