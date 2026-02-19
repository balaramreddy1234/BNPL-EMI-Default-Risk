def validate_csv(df):
    required = ["income","loan","emi","tenure","ontime","delays","credit"]
    return all(col in df.columns for col in required)
