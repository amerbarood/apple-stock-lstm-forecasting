# Apple Stock Price Forecasting with LSTM, BiLSTM and GRU

Predicting Apple's next-day stock price from the previous 30 days, comparing three recurrent neural network architectures, following the CRISP-DM process.

> Group project for *Advanced Data Mining (D7043E)*, Luleå University of Technology. Group 17.

## Highlights

- **LSTM won clearly:** test RMSE of 7.34, versus 13.35 for BiLSTM and 13.74 for GRU.
- **Full CRISP-DM workflow:** business understanding, cleaning of a deliberately messy dataset, feature engineering, modelling and evaluation.
- **One script compares all three models** with identical data, splits and training settings.

## Problem

Stock prices are noisy, non-linear time series, and simple models struggle to capture their temporal dependencies. The goal was to forecast Apple's next-day price and find out which recurrent architecture handles this best.

## Data

`AAPL_messy.xlsx`, a course-provided dataset of 1,259 daily records combining Apple's share price with financial, operational and market indicators (brand value, R&D investment, marketing spend, customer satisfaction, smartphone demand, market sentiment, interest and exchange rates, product category, region).

As the name suggests, it is intentionally dirty: missing values, outliers such as an R&D figure of 24,171.6, mixed data types, redundant columns and date gaps. The dataset is not redistributed here; place it at `data/AAPL_messy.xlsx` to run the code.

## Approach

**Cleaning ([`src/data.py`](src/data.py))**
- Parsed and sorted dates, and dropped redundant columns (`Price_Copy`, `Random Notes`).
- Interpolated missing numeric values over time and filled categories with the mode.
- Engineered `Price Change` and an `R&D/Marketing Ratio`, with optional IQR outlier removal for exploration.

**Exploration ([`src/eda.py`](src/eda.py))**
Price over time, correlation heatmap, price by market sentiment and product category, and a pair plot of key business drivers.

**Modelling ([`src/models.py`](src/models.py), [`src/train.py`](src/train.py))**
- Price scaled to [0, 1] and turned into 30-day lookback windows.
- Chronological split: 70% train, 15% validation, 15% test, so the model never sees the future.
- Same structure for every model: two recurrent layers (64 → 32 units), 20% dropout, a 16-unit dense layer and a linear output.
- Adam optimiser, MSE loss, 50 epochs, batch size 32.

## Results

| Model | Test RMSE ($) | Training time |
|---|---|---|
| **LSTM** | **7.34** | Medium |
| BiLSTM | 13.35 | Long |
| GRU | 13.74 | Fast |

**Why LSTM won:** it captured the sequential pattern with the closest, least laggy fit. The bidirectional model adds complexity that doesn't help when forecasting forward in time, and the GRU trained fastest but tracked longer-range movements less accurately.

## Limitations and next steps

- **Price-only input.** The models currently use only the price history. Feeding in the engineered business and macro features would test whether they really add signal.
- **A naive baseline is needed.** For next-day prices, "tomorrow equals today" is a strong benchmark, and comparing against it would show how much the networks actually learn.
- **Single run per model.** Repeating training with several random seeds would show how stable the ranking is.

## Repository structure

```
├── src/
│   ├── data.py     # loading, cleaning, feature engineering, windowing
│   ├── models.py   # LSTM, BiLSTM and GRU definitions
│   ├── train.py    # train and evaluate one or all models
│   └── eda.py      # exploratory plots
├── data/           # put AAPL_messy.xlsx here (not included)
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
cd src
python train.py --data ../data/AAPL_messy.xlsx --model all
python eda.py --data ../data/AAPL_messy.xlsx
```

The code was refactored from the original Colab notebook into reusable modules, keeping the same preprocessing, architectures and training settings.

## Team

**Aamer Barood** (Aamer Ahmed) and Hind Dafallah.
