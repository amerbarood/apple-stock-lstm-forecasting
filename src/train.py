"""Train and evaluate LSTM, BiLSTM or GRU on next-day AAPL price prediction.

Usage:
    python src/train.py --data data/AAPL_messy.xlsx --model lstm
    python src/train.py --data data/AAPL_messy.xlsx --model all
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import load_and_clean, prepare_price_windows
from models import build_model


def run(kind: str, df, seq_len: int, epochs: int, batch_size: int, plot: bool):
    splits, scaler = prepare_price_windows(df, seq_len)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = splits["train"], splits["val"], splits["test"]

    model = build_model(kind, seq_len)
    start = time.time()
    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, verbose=0)
    train_seconds = time.time() - start

    y_true = scaler.inverse_transform(y_te)
    y_pred = scaler.inverse_transform(model.predict(X_te, verbose=0))
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "train_seconds": round(train_seconds, 1),
    }

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
        ax1.plot(y_true, label="Actual", linewidth=2)
        ax1.plot(y_pred, label="Predicted", linestyle="--")
        ax1.set(title=f"{kind.upper()}: test set", xlabel="Day", ylabel="Price ($)")
        ax1.legend()
        ax2.plot(history.history["loss"], label="Train")
        ax2.plot(history.history["val_loss"], label="Validation")
        ax2.set(title="Loss (MSE)", xlabel="Epoch")
        ax2.legend()
        plt.tight_layout()
        os.makedirs("assets", exist_ok=True)
        plt.savefig(f"assets/{kind}_results.png", dpi=130)
        plt.close(fig)

    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/AAPL_messy.xlsx")
    p.add_argument("--model", default="all", choices=["lstm", "bilstm", "gru", "all"])
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    df = load_and_clean(args.data)
    kinds = ["lstm", "bilstm", "gru"] if args.model == "all" else [args.model]
    for kind in kinds:
        m = run(kind, df, args.seq_len, args.epochs, args.batch_size, not args.no_plot)
        print(f"{kind:7s} RMSE {m['RMSE']:.2f}  MAE {m['MAE']:.2f}  R2 {m['R2']:.3f}  ({m['train_seconds']}s)")


if __name__ == "__main__":
    main()
