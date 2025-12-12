#!/usr/bin/env python3
"""
Run Chronos-2 zero-shot inference on a time series dataset and visualize results.

Default behavior:
  - Download Chronos-2 weights from Hugging Face ("amazon/chronos-2")
  - Download the electricity price demo dataset (train/test parquet) from AutoGluon S3
  - Forecast the first time series and save a plot

You can also pass local parquet files to avoid network access.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

from chronos import Chronos2Pipeline


DEFAULT_CONTEXT_URL = "https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/train.parquet"
DEFAULT_FUTURE_URL = "https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/test.parquet"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chronos-2 inference + visualization")
    parser.add_argument("--model-id", default="amazon/chronos-2", help="Hugging Face model repo id")
    parser.add_argument("--device", default=None, help='Device map, e.g. "cuda", "cpu", or "auto"')
    parser.add_argument(
        "--context-parquet",
        default=None,
        help="Local parquet for historical context. If omitted, downloads demo dataset.",
    )
    parser.add_argument(
        "--future-parquet",
        default=None,
        help="Local parquet for future covariates + ground truth. If omitted, downloads demo dataset.",
    )
    parser.add_argument("--prediction-length", type=int, default=24, help="Forecast horizon")
    parser.add_argument(
        "--quantiles",
        default="0.1,0.5,0.9",
        help="Comma-separated quantile levels for probabilistic forecasts.",
    )
    parser.add_argument("--id-column", default="id", help="Series id column name")
    parser.add_argument("--timestamp-column", default="timestamp", help="Timestamp column name")
    parser.add_argument("--target-column", default="target", help="Target column name")
    parser.add_argument(
        "--series-id",
        default=None,
        help="Which series id to plot. Defaults to first id in dataset.",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=256,
        help="How many historical points to show in plot.",
    )
    parser.add_argument(
        "--outdir",
        default="outputs",
        help="Directory to save plot and predictions.",
    )
    parser.add_argument(
        "--no-future-covariates",
        action="store_true",
        help="Ignore future covariates even if present.",
    )
    return parser.parse_args(argv)


def load_parquet(local_path: str | None, url: str) -> pd.DataFrame:
    if local_path is not None:
        return pd.read_parquet(local_path)
    return pd.read_parquet(url)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model {args.model_id!r} on device={device!r} ...")
    pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map=device)

    print("Loading dataset ...")
    context_df = load_parquet(args.context_parquet, DEFAULT_CONTEXT_URL)
    future_df_full = load_parquet(args.future_parquet, DEFAULT_FUTURE_URL)

    if args.no_future_covariates:
        future_df = None
        test_df = future_df_full
    else:
        test_df = future_df_full
        future_df = test_df.drop(columns=[args.target_column])

    quantiles = [float(q.strip()) for q in args.quantiles.split(",") if q.strip()]

    print("Running prediction ...")
    pred_df = pipeline.predict_df(
        context_df,
        future_df=future_df,
        prediction_length=args.prediction_length,
        quantile_levels=quantiles,
        id_column=args.id_column,
        timestamp_column=args.timestamp_column,
        target=args.target_column,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pred_path = outdir / "predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")

    # Pick a series to plot
    series_ids = context_df[args.id_column].unique()
    if len(series_ids) == 0:
        raise ValueError("No series found in context_df")
    series_id = series_ids[0] if args.series_id is None else args.series_id

    ctx_one = context_df[context_df[args.id_column] == series_id].copy()
    test_one = test_df[test_df[args.id_column] == series_id].copy()
    pred_one = pred_df[pred_df[args.id_column] == series_id].copy()

    ctx_one = ctx_one.sort_values(args.timestamp_column)
    test_one = test_one.sort_values(args.timestamp_column)
    pred_one = pred_one.sort_values(args.timestamp_column)

    ctx_series = ctx_one.set_index(args.timestamp_column)[args.target_column].tail(args.history_length)
    gt_series = test_one.set_index(args.timestamp_column)[args.target_column]
    pred_series = pred_one.set_index(args.timestamp_column)["predictions"]

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for visualization: pip install matplotlib") from exc

    plt.figure(figsize=(12, 3))
    ctx_series.plot(label="historical data", color="tab:blue")
    gt_series.plot(label="future data (ground truth)", color="tab:green")
    pred_series.plot(label="forecast (median)", color="tab:purple")

    q_low = min(quantiles)
    q_high = max(quantiles)
    if f"{q_low:g}" in pred_one.columns and f"{q_high:g}" in pred_one.columns:
        low = pred_one.set_index(args.timestamp_column)[f"{q_low:g}"]
        high = pred_one.set_index(args.timestamp_column)[f"{q_high:g}"]
        plt.fill_between(pred_series.index, low, high, alpha=0.3, label=f"{q_low}-{q_high} interval")

    plt.title(f"Chronos-2 forecast for series id={series_id}")
    plt.legend()
    plt.tight_layout()

    fig_path = outdir / "forecast.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved plot to {fig_path}")


if __name__ == "__main__":
    main()

