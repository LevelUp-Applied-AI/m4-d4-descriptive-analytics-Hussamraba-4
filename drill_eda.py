"""Core Skills Drill — Descriptive Analytics

Compute summary statistics, plot distributions, and create a correlation
heatmap for the sample sales dataset.

Usage:
    python drill_eda.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def compute_summary(df):
    """Compute summary statistics for all numeric columns.

    Args:
        df: pandas DataFrame with at least some numeric columns

    Returns:
        DataFrame containing count, mean, median, std, min, max
        for each numeric column. Save the result to output/summary.csv.
    """
    os.makedirs("output", exist_ok=True)

    numeric_df = df.select_dtypes(include=[np.number])

    summary = pd.DataFrame({
        "count": numeric_df.count(),
        "mean": numeric_df.mean(),
        "median": numeric_df.median(),
        "std": numeric_df.std(),
        "min": numeric_df.min(),
        "max": numeric_df.max()
    }).T

    summary.to_csv("output/summary.csv")
    return summary


def plot_distributions(df, columns, output_path):
    """Create a 2x2 subplot figure with histograms for the specified columns.

    Args:
        df: pandas DataFrame
        columns: list of 4 column names to plot (use numeric columns)
        output_path: file path to save the figure (e.g., 'output/distributions.png')

    Returns:
        None — saves the figure to output_path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    ax = ax.flatten()

    for i, col in enumerate(columns):
        sns.histplot(df[col], kde=True, ax=ax[i])
        ax[i].set_title(f"Distribution of {col}")
        ax[i].set_xlabel(col)
        ax[i].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_correlation(df, output_path):
    """Compute Pearson correlation matrix and visualize as a heatmap.

    Args:
        df: pandas DataFrame with numeric columns
        output_path: file path to save the figure (e.g., 'output/correlation.png')

    Returns:
        None — saves the figure to output_path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(method="pearson")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    """Load data, compute summary, and generate all plots."""
    os.makedirs("output", exist_ok=True)

    df = pd.read_csv("data/sample_sales.csv")

    compute_summary(df)

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_columns = numeric_columns[:4]

    plot_distributions(df, selected_columns, "output/distributions.png")
    plot_correlation(df, "output/correlation.png")


if __name__ == "__main__":
    main()