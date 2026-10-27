"""Loading, cleaning and windowing of the AAPL dataset."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DROP_COLUMNS = ["Price_Copy", "Random Notes"]


def load_and_clean(path: str, remove_outliers: bool = False) -> pd.DataFrame:
    """Load the raw Excel file and return a clean, date-sorted DataFrame."""
    df = pd.read_excel(path, sheet_name="Sheet1")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    df = df.dropna(subset=["Date", "Price"]).sort_values("Date")

    # Fill gaps: interpolate numbers over time, use the mode for categories
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Feature engineering
    df["Price Change"] = df["Price"].diff()
    df["R&D/Marketing Ratio"] = df["R&D Investment"] / df["Marketing Spend"]

    if remove_outliers:
        for col in numeric_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]

    return df.reset_index(drop=True)


def make_sequences(values: np.ndarray, seq_len: int):
    """Turn a series into (lookback window, next value) pairs."""
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i : i + seq_len])
        y.append(values[i + seq_len])
    return np.array(X), np.array(y)


def prepare_price_windows(df: pd.DataFrame, seq_len: int = 30,
                          train_frac: float = 0.70, val_frac: float = 0.15):
    """Scale the price to [0, 1], window it and split chronologically."""
    prices = df.sort_values("Date").set_index("Date")[["Price"]]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X, y = make_sequences(scaled, seq_len)
    n_train = int(train_frac * len(X))
    n_val = int(val_frac * len(X))

    splits = {
        "train": (X[:n_train], y[:n_train]),
        "val": (X[n_train : n_train + n_val], y[n_train : n_train + n_val]),
        "test": (X[n_train + n_val :], y[n_train + n_val :]),
    }
    return splits, scaler
