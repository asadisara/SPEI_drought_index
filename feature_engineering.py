import pandas as pd

def create_features(df, lags=12):
    df = df.copy()
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['Average'].shift(i)

    df['rolling_mean_3'] = df['Average'].rolling(window=3).mean()
    df['rolling_std_3'] = df['Average'].rolling(window=3).std()
    df['month'] = df.index.month

    df.dropna(inplace=True)
    return df
