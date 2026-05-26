import os
import re
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr


COLUMN_ORDER = [
    "model",
    "layer",

    "human_ent_mean_r2",
    "human_ent_fold_r2s",
    "human_ent_mean_correlation",
    "human_ent_fold_correlations",
    "human_ent_mean_mse",
    "human_ent_fold_mses",

    "rt_correct_mean_r2",
    "rt_correct_fold_r2s",
    "rt_correct_mean_correlation",
    "rt_correct_fold_correlations",
    "rt_correct_mean_mse",
    "rt_correct_fold_mses",

    "rt_combined_mean_r2",
    "rt_combined_fold_r2s",
    "rt_combined_mean_correlation",
    "rt_combined_fold_correlations",
    "rt_combined_mean_mse",
    "rt_combined_fold_mses",

    "n_samples",
    "n_features"
]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--activation_dir",
        type=str,
        default="/work/projects/bs-wdward43/wdward43/activation_outputs/"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/work/projects/bs-wdward43/wdward43/"
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default="uncertainty_alignment_10fold_linear_probe_results"
    )

    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--progress_log", type=str, default=None)

    return parser.parse_args()


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message, progress_log=None):
    line = f"[{timestamp()}] {message}"
    print(line, flush=True)

    if progress_log is not None:
        progress_log.parent.mkdir(parents=True, exist_ok=True)

        with progress_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


def get_corr(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    try:
        corr, _ = pearsonr(y_true, y_pred)
        return float(corr)
    except Exception:
        return np.nan


def save_partial_results(rows, output_dir, output_name, progress_log=None):
    if len(rows) == 0:
        return

    results_df = pd.DataFrame(rows)

    for col in COLUMN_ORDER:
        if col not in results_df.columns:
            results_df[col] = np.nan

    results_df = results_df[COLUMN_ORDER]

    results_df["layer_num"] = (
        results_df["layer"]
        .str.replace("layer_", "", regex=False)
        .astype(int)
    )

    results_df = results_df.sort_values(["model", "layer_num"])
    results_df = results_df.drop(columns=["layer_num"])

    output_dir.mkdir(parents=True, exist_ok=True)

    partial_csv_out = output_dir / f"{output_name}_PARTIAL.csv"
    partial_parquet_out = output_dir / f"{output_name}_PARTIAL.parquet"

    results_df.to_csv(partial_csv_out, index=False)
    results_df.to_parquet(partial_parquet_out, index=False)

    log(f"Partial save: {len(results_df)} rows", progress_log)


def run_probe(
    df,
    target_col,
    n_splits=10,
    random_state=42
):
    dim_cols = [c for c in df.columns if str(c).startswith("dim_")]

    if len(dim_cols) == 0:
        raise ValueError("No dim columns found")

    X = df[dim_cols].to_numpy(dtype=np.float32)
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float32)

    keep = np.isfinite(y)
    X = X[keep]
    y = y[keep]

    if len(y) < n_splits:
        raise ValueError(f"Only {len(y)} valid rows for {target_col}")

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    r2s = []
    corrs = []
    mses = []

    for train_idx, test_idx in kf.split(X):
        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        reg = LinearRegression()
        reg.fit(X_train, y_train)

        preds = reg.predict(X_test)

        r2s.append(float(r2_score(y_test, preds)))
        corrs.append(float(get_corr(y_test, preds)))
        mses.append(float(mean_squared_error(y_test, preds)))

    return {
        "mean_r2": float(np.nanmean(r2s)),
        "fold_r2s": r2s,

        "mean_correlation": float(np.nanmean(corrs)),
        "fold_correlations": corrs,

        "mean_mse": float(np.nanmean(mses)),
        "fold_mses": mses,

        "n_samples": int(len(y)),
        "n_features": int(X.shape[1])
    }


def main():
    args = get_args()

    activation_dir = Path(args.activation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_log = (
        Path(args.progress_log)
        if args.progress_log is not None
        else output_dir / f"{args.output_name}_progress.log"
    )

    target_map = {
        "human-ent": "human_ent",
        "MC_Human_RT_Correct": "rt_correct",
        "MC_Human_RT_Combined": "rt_combined"
    }

    rows = []

    log("Starting 10-fold linear probes", progress_log)
    log(f"activation_dir={activation_dir}", progress_log)
    log(f"output_dir={output_dir}", progress_log)

    model_folders = [p for p in sorted(activation_dir.iterdir()) if p.is_dir()]

    total_layers = 0
    layer_counts = {}

    for model_folder in model_folders:
        model_name = model_folder.name
        pattern = re.compile(rf"{re.escape(model_name)}-layer-(\d+)\.parquet")
        count = 0

        for file_path in model_folder.iterdir():
            if pattern.match(file_path.name):
                count += 1

        layer_counts[model_name] = count
        total_layers += count

    log(f"Found {len(model_folders)} models and {total_layers} layer files", progress_log)

    completed_layers = 0

    for model_idx, model_folder in enumerate(model_folders, start=1):
        model_name = model_folder.name
        pattern = re.compile(rf"{re.escape(model_name)}-layer-(\d+)\.parquet")

        layer_files = []

        for file_path in model_folder.iterdir():
            match = pattern.match(file_path.name)

            if match:
                layer_num = int(match.group(1))
                layer_files.append((layer_num, file_path))

        layer_files = sorted(layer_files, key=lambda x: x[0])

        log(
            f"Model {model_idx}/{len(model_folders)}: {model_name} "
            f"({len(layer_files)} layers)",
            progress_log
        )

        for layer_idx, (layer_num, file_path) in enumerate(layer_files, start=1):
            completed_layers += 1

            log(
                f"{model_name} layer {layer_num} "
                f"({layer_idx}/{len(layer_files)}, overall {completed_layers}/{total_layers})",
                progress_log
            )

            df = pd.read_parquet(file_path)

            row = {
                "model": model_name,
                "layer": f"layer_{layer_num}",
                "n_samples": np.nan,
                "n_features": np.nan
            }

            for target_col, short_name in target_map.items():
                try:
                    metrics = run_probe(
                        df,
                        target_col,
                        n_splits=args.n_splits,
                        random_state=args.random_state
                    )

                    row[f"{short_name}_mean_r2"] = metrics["mean_r2"]
                    row[f"{short_name}_fold_r2s"] = metrics["fold_r2s"]

                    row[f"{short_name}_mean_correlation"] = metrics["mean_correlation"]
                    row[f"{short_name}_fold_correlations"] = metrics["fold_correlations"]

                    row[f"{short_name}_mean_mse"] = metrics["mean_mse"]
                    row[f"{short_name}_fold_mses"] = metrics["fold_mses"]

                    row["n_samples"] = metrics["n_samples"]
                    row["n_features"] = metrics["n_features"]

                    log(
                        f"  {target_col}: "
                        f"r2={metrics['mean_r2']:.4f}, "
                        f"corr={metrics['mean_correlation']:.4f}, "
                        f"mse={metrics['mean_mse']:.4f}",
                        progress_log
                    )

                except Exception as e:
                    log(f"  {target_col}: skipped ({e})", progress_log)

                    row[f"{short_name}_mean_r2"] = np.nan
                    row[f"{short_name}_fold_r2s"] = [np.nan] * args.n_splits

                    row[f"{short_name}_mean_correlation"] = np.nan
                    row[f"{short_name}_fold_correlations"] = [np.nan] * args.n_splits

                    row[f"{short_name}_mean_mse"] = np.nan
                    row[f"{short_name}_fold_mses"] = [np.nan] * args.n_splits

            rows.append(row)

            save_partial_results(rows, output_dir, args.output_name, progress_log)

            del df

        log(f"Finished {model_name}", progress_log)

    results_df = pd.DataFrame(rows)

    for col in COLUMN_ORDER:
        if col not in results_df.columns:
            results_df[col] = np.nan

    results_df = results_df[COLUMN_ORDER]

    results_df["layer_num"] = (
        results_df["layer"]
        .str.replace("layer_", "", regex=False)
        .astype(int)
    )

    results_df = results_df.sort_values(["model", "layer_num"])
    results_df = results_df.drop(columns=["layer_num"])

    parquet_out = output_dir / f"{args.output_name}.parquet"
    csv_out = output_dir / f"{args.output_name}.csv"

    results_df.to_parquet(parquet_out, index=False)
    results_df.to_csv(csv_out, index=False)

    log(f"Saved parquet: {parquet_out}", progress_log)
    log(f"Saved csv: {csv_out}", progress_log)
    log("Done", progress_log)


if __name__ == "__main__":
    main()