"""The three recurrent architectures compared in the project."""
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.models import Sequential


def build_model(kind: str, seq_len: int = 30) -> Sequential:
    """Two stacked recurrent layers (64 -> 32 units) with dropout, then a dense head."""
    kind = kind.lower()
    if kind == "lstm":
        rnn1, rnn2 = LSTM(64, return_sequences=True), LSTM(32)
    elif kind == "bilstm":
        rnn1 = Bidirectional(LSTM(64, return_sequences=True))
        rnn2 = Bidirectional(LSTM(32))
    elif kind == "gru":
        rnn1, rnn2 = GRU(64, return_sequences=True), GRU(32)
    else:
        raise ValueError(f"Unknown model '{kind}', use lstm, bilstm or gru")

    model = Sequential([
        Input(shape=(seq_len, 1)),
        rnn1,
        Dropout(0.2),
        rnn2,
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model
