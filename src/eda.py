"""Exploratory plots: price over time, correlations, sentiment and category views."""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data import load_and_clean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/AAPL_messy.xlsx")
    args = p.parse_args()

    df = load_and_clean(args.data, remove_outliers=True)
    sns.set(style="whitegrid")

    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Price"], label="Price", linewidth=2)
    plt.plot(df["Date"], df["Next Day Price"], label="Next Day Price", linestyle="--")
    plt.title("Apple stock price over time")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.show()

    numeric = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric.corr(), cmap="coolwarm")
    plt.title("Correlation between financial metrics")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Market Sentiment", y="Price", data=df)
    plt.title("Price by market sentiment")
    plt.show()

    plt.figure(figsize=(10, 5))
    avg = df.groupby("Product Category")["Price"].mean().sort_values(ascending=False)
    sns.barplot(x=avg.index, y=avg.values, palette="viridis")
    plt.title("Average price by product category")
    plt.xticks(rotation=45)
    plt.show()

    sns.pairplot(df[["Price", "Brand Value", "R&D Investment", "Marketing Spend",
                     "Customer Satisfaction"]])
    plt.suptitle("Relationships between key variables", y=1.02)
    plt.show()


if __name__ == "__main__":
    main()
