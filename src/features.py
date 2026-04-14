import pandas as pd

def create_features(df):
    df = df.copy()

    df["temp_roll_mean"] = df["temperature"].rolling(5).mean()
    df["vib_roll_mean"] = df["vibration"].rolling(5).mean()
    df["rpm_roll_mean"] = df["rpm"].rolling(5).mean()

    df = df.dropna()
    return df