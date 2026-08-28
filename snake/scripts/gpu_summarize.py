#!/usr/bin/env python
# snake/scripts/gpu_summarize.py
#
# Summarize an nvidia-smi CSV log into a short human-readable report.
#
# Reads the raw CSV produced by:
#   nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
#              --format=csv -l <interval> > gpu_raw.csv
#
# Writes mean / median / p90 / p99 GPU utilization and peak memory. The whole distribution
# is reported, not just the mean: on small models the mean is pulled down by legitimate CPU
# phases (data load, per-slide metric computation), so a low mean with a high p90 is
# healthy. A low p90 on a multi-gene run is the warning sign.
#
# Fail-soft: a missing, empty or malformed CSV (e.g. no GPU on the node) writes a note
# instead of raising, so it never fails the training job that calls it.
#
# Usage:
#   python gpu_summarize.py gpu_raw.csv gpu_summary.txt [--device N] [--skip-seconds S]

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_csv")
    ap.add_argument("out_txt")
    ap.add_argument("--device", type=int, default=None,
                    help="Only summarize this GPU index (default: all rows).")
    ap.add_argument("--skip-seconds", type=float, default=0.0,
                    help="Drop the first S seconds (the CPU-bound data-load phase) "
                         "so the summary reflects training, not startup.")
    a = ap.parse_args()

    def write(msg):
        with open(a.out_txt, "w") as f:
            f.write(msg + "\n")

    try:
        import pandas as pd
        import numpy as np
    except Exception as e:                       # pragma: no cover
        write(f"gpu_summarize: pandas/numpy unavailable ({e}); no summary.")
        return

    try:
        df = pd.read_csv(a.raw_csv, skipinitialspace=True)
    except Exception as e:
        write(f"gpu_summarize: could not read {a.raw_csv} ({e}); no GPU data logged.")
        return

    if len(df) == 0:
        write(f"gpu_summarize: {a.raw_csv} is empty; no GPU data logged.")
        return

    # nvidia-smi columns look like "utilization.gpu [%]"; normalize names.
    def find(col_substr):
        for c in df.columns:
            if col_substr in c:
                return c
        return None

    util_col = find("utilization.gpu")
    mem_used_col = find("memory.used")
    idx_col = find("index")
    ts_col = find("timestamp")

    if util_col is None:
        write(f"gpu_summarize: no utilization column in {a.raw_csv}; columns={list(df.columns)}")
        return

    # Optional: restrict to one device (multi-GPU nodes report all visible GPUs).
    if a.device is not None and idx_col is not None:
        df = df[df[idx_col].astype(str).str.strip() == str(a.device)]
        if len(df) == 0:
            write(f"gpu_summarize: no rows for device {a.device}.")
            return

    # Values come as strings like "12 %" or "1234 MiB"; strip units to numbers.
    def to_num(series):
        return pd.to_numeric(
            series.astype(str).str.replace("%", "", regex=False)
                              .str.replace("MiB", "", regex=False)
                              .str.strip(),
            errors="coerce",
        )

    util = to_num(df[util_col]).dropna()

    # Optionally drop the first S seconds (startup/load phase).
    if a.skip_seconds > 0 and ts_col is not None:
        try:
            t = pd.to_datetime(df[ts_col], errors="coerce")
            keep = t >= (t.iloc[0] + pd.Timedelta(seconds=a.skip_seconds))
            util = to_num(df.loc[keep, util_col]).dropna()
        except Exception:
            pass                                 # if timestamps unparseable, keep all

    if len(util) == 0:
        write("gpu_summarize: no numeric utilization samples after filtering.")
        return

    lines = [
        f"GPU utilization summary  ({len(util)} samples"
        + (f", first {a.skip_seconds:.0f}s skipped" if a.skip_seconds else "") + ")",
        f"  mean   : {util.mean():6.1f} %",
        f"  median : {util.median():6.1f} %",
        f"  p90    : {np.percentile(util, 90):6.1f} %",
        f"  p99    : {np.percentile(util, 99):6.1f} %",
        f"  max    : {util.max():6.1f} %",
        f"  min    : {util.min():6.1f} %",
    ]

    if mem_used_col is not None:
        mem = to_num(df[mem_used_col]).dropna()
        if len(mem):
            lines.append(f"  peak memory used : {mem.max():.0f} MiB")

    lines.append("")
    lines.append("Reading this: low mean + high p90 is HEALTHY — the GPU saturates")
    lines.append("when there is work; the mean is pulled down by legitimate CPU phases")
    lines.append("(data load, per-slide metric computation). Low p90 on a MULTI-gene")
    lines.append("run is the real warning sign (feeding pipeline not keeping up).")

    write("\n".join(lines))
    print("\n".join(lines))                       # also echo to the job log


if __name__ == "__main__":
    main()
