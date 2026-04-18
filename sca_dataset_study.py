from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold


sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120


@dataclass
class SampleRecord:
    sample_id: str
    folder_id: str
    split: str
    subset: str
    file_path: str
    sample_idx: int
    sampling_rate: float
    rpm: float
    label: int
    signal_length: int
    fault_ftf_mult: float
    fault_bsf_mult: float
    fault_bpfo_mult: float
    fault_bpfi_mult: float
    rms: float
    kurtosis: float
    skewness: float
    envelope_kurtosis: float
    E_bpfi: float
    E_bpfo: float
    E_bsf: float
    nan_count: int
    inf_count: int
    signal_hash: str


def _safe_float(x) -> float:
    arr = np.asarray(x).squeeze()
    if arr.size == 0:
        return np.nan
    return float(arr)


def _mat_fault_multipliers(fault_frequencies) -> Dict[str, float]:
    return {
        "FTF": _safe_float(getattr(fault_frequencies, "FTFMultiple", np.nan)),
        "BSF": _safe_float(getattr(fault_frequencies, "BPFMultiple", np.nan)),
        "BPFO": _safe_float(getattr(fault_frequencies, "BPFOMultiple", np.nan)),
        "BPFI": _safe_float(getattr(fault_frequencies, "BPFIMultiple", np.nan)),
    }


def _signal_hash(signal: np.ndarray) -> str:
    rounded = np.round(signal.astype(np.float64), 8)
    return hashlib.md5(rounded.tobytes()).hexdigest()


def _fft_energy_around(
    signal: np.ndarray,
    fs: float,
    target_hz: float,
    bandwidth_hz: float = 3.0,
) -> float:
    x = signal - np.mean(signal)
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(len(x), d=1.0 / fs)
    mask = (freq >= target_hz - bandwidth_hz) & (freq <= target_hz + bandwidth_hz)
    if not np.any(mask):
        return 0.0
    return float(np.sum(spectrum[mask]))


def _envelope(signal: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(signal - np.mean(signal)))


def _envelope_spectrum(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    env = _envelope(signal)
    env = env - np.mean(env)
    spec = np.abs(np.fft.rfft(env))
    freq = np.fft.rfftfreq(len(env), d=1.0 / fs)
    return freq, spec


def _extract_signals(raw_data) -> List[np.ndarray]:
    arr = np.asarray(raw_data)
    if arr.dtype == object:
        return [np.asarray(x, dtype=np.float64).reshape(-1) for x in arr.reshape(-1)]
    if arr.ndim == 1:
        return [arr.astype(np.float64).reshape(-1)]
    if arr.ndim == 2:
        return [arr[i, :].astype(np.float64).reshape(-1) for i in range(arr.shape[0])]
    reshaped = arr.reshape(arr.shape[0], -1)
    return [reshaped[i, :].astype(np.float64).reshape(-1) for i in range(reshaped.shape[0])]


def _align_vector(v, n: int, dtype=float) -> np.ndarray:
    arr = np.asarray(v, dtype=dtype).reshape(-1)
    if arr.size == n:
        return arr
    if arr.size == 1:
        return np.full(n, arr.item(), dtype=dtype)
    # Conservative fallback: trim or pad with last value.
    if arr.size > n:
        return arr[:n]
    out = np.empty(n, dtype=dtype)
    out[: arr.size] = arr
    out[arr.size :] = arr[-1]
    return out


def collect_dataset_records(dataset_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[SampleRecord] = []
    file_rows = []

    mat_files = sorted(dataset_root.glob("*/*.mat"))
    for mat_path in mat_files:
        folder_id = mat_path.parent.name
        split = mat_path.stem
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

        for subset in ["DS", "FS", "Upper", "Lower"]:
            if subset not in mat:
                continue
            struct = mat[subset]
            signals = _extract_signals(struct.rawData)
            n_samples = len(signals)
            sr = _align_vector(struct.samplingRate, n_samples, dtype=np.float64)
            rpm = _align_vector(struct.RPM, n_samples, dtype=np.float64)
            labels = _align_vector(struct.label, n_samples, dtype=np.int32)
            multipliers = _mat_fault_multipliers(struct.faultFrequencies)

            lengths = np.array([len(s) for s in signals], dtype=int)

            file_rows.append(
                {
                    "folder_id": folder_id,
                    "split": split,
                    "subset": subset,
                    "file_path": str(mat_path),
                    "n_samples": n_samples,
                    "signal_length_min": int(lengths.min()),
                    "signal_length_max": int(lengths.max()),
                    "sampling_rate_mean": float(np.mean(sr)),
                    "sampling_rate_unique": int(np.unique(sr).size),
                    "rpm_mean": float(np.mean(rpm)),
                    "healthy_count": int(np.sum(labels == 0)),
                    "faulty_count": int(np.sum(labels == -1)),
                }
            )

            for i, sig in enumerate(signals):
                fs_i = float(sr[i])
                rpm_i = float(rpm[i])

                # Assume multipliers are fault orders and convert to Hz with shaft speed (RPM / 60).
                shaft_hz = rpm_i / 60.0 if rpm_i > 0 else np.nan
                bpfi_hz = multipliers["BPFI"] * shaft_hz if np.isfinite(shaft_hz) else np.nan
                bpfo_hz = multipliers["BPFO"] * shaft_hz if np.isfinite(shaft_hz) else np.nan
                bsf_hz = multipliers["BSF"] * shaft_hz if np.isfinite(shaft_hz) else np.nan

                env = _envelope(sig)
                rows.append(
                    SampleRecord(
                        sample_id=f"{folder_id}_{split}_{subset}_{i}",
                        folder_id=folder_id,
                        split=split,
                        subset=subset,
                        file_path=str(mat_path),
                        sample_idx=i,
                        sampling_rate=fs_i,
                        rpm=rpm_i,
                        label=int(labels[i]),
                        signal_length=int(len(sig)),
                        fault_ftf_mult=multipliers["FTF"],
                        fault_bsf_mult=multipliers["BSF"],
                        fault_bpfo_mult=multipliers["BPFO"],
                        fault_bpfi_mult=multipliers["BPFI"],
                        rms=float(np.sqrt(np.mean(sig ** 2))),
                        kurtosis=float(kurtosis(sig, fisher=False, bias=False)),
                        skewness=float(skew(sig, bias=False)),
                        envelope_kurtosis=float(kurtosis(env, fisher=False, bias=False)),
                        E_bpfi=_fft_energy_around(sig, fs_i, bpfi_hz) if np.isfinite(bpfi_hz) else 0.0,
                        E_bpfo=_fft_energy_around(sig, fs_i, bpfo_hz) if np.isfinite(bpfo_hz) else 0.0,
                        E_bsf=_fft_energy_around(sig, fs_i, bsf_hz) if np.isfinite(bsf_hz) else 0.0,
                        nan_count=int(np.isnan(sig).sum()),
                        inf_count=int(np.isinf(sig).sum()),
                        signal_hash=_signal_hash(sig),
                    )
                )

    records_df = pd.DataFrame([r.__dict__ for r in rows])
    files_df = pd.DataFrame(file_rows)
    return records_df, files_df


def _class_name(y: int) -> str:
    if y == 0:
        return "healthy"
    if y == -1:
        return "faulty"
    return f"other_{y}"


def plot_label_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df["label"].map(_class_name).value_counts()
    order = ["healthy", "faulty"] + sorted([c for c in counts.index if c.startswith("other_")])
    counts = counts.reindex(order).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(6, 4))
    palette = ["#2a9d8f", "#e76f51"] + ["#999999"] * max(0, len(counts) - 2)
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=palette, legend=False, ax=ax)
    ax.set_title("Overall Label Distribution")
    ax.set_ylabel("Sample Count")
    for i, val in enumerate(counts.values):
        ax.text(i, val, str(val), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "label_distribution_overall.png")
    plt.close(fig)

    grouped = (
        df.assign(class_name=df["label"].map(_class_name))
        .groupby(["folder_id", "split", "subset", "class_name"])
        .size()
        .reset_index(name="count")
    )
    grouped["file_subset"] = grouped["folder_id"] + "_" + grouped["split"] + "_" + grouped["subset"]
    fig, ax = plt.subplots(figsize=(16, 5))
    sns.barplot(data=grouped, x="file_subset", y="count", hue="class_name", ax=ax)
    ax.set_title("Per File Label Distribution")
    ax.set_xlabel("Folder_Split_Subset")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(output_dir / "label_distribution_per_file.png")
    plt.close(fig)


def plot_rpm_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df, x="rpm", hue=df["label"].map(_class_name), bins=40, kde=True, ax=ax)
    ax.set_title("RPM Distribution by Label")
    fig.tight_layout()
    fig.savefig(output_dir / "rpm_distribution_by_label.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df.assign(class_name=df["label"].map(_class_name)), x="class_name", y="rpm", ax=ax)
    ax.set_title("RPM vs Label")
    fig.tight_layout()
    fig.savefig(output_dir / "rpm_vs_label_boxplot.png")
    plt.close(fig)


def plot_time_domain_examples(
    dataset_root: Path,
    df: pd.DataFrame,
    output_dir: Path,
    n_examples: int = 3,
) -> None:
    examples = []
    for label in [0, -1]:
        class_df = df[df["label"] == label].head(n_examples)
        examples.extend(class_df.to_dict("records"))

    fig, axes = plt.subplots(len(examples), 1, figsize=(14, 2.2 * len(examples)), sharex=False)
    if len(examples) == 1:
        axes = [axes]

    for ax, row in zip(axes, examples):
        mat = loadmat(row["file_path"], squeeze_me=True, struct_as_record=False)
        struct = mat[row["subset"]]
        sig = np.asarray(struct.rawData[row["sample_idx"]], dtype=np.float64)
        fs = float(np.asarray(struct.samplingRate).reshape(-1)[row["sample_idx"]])
        t = np.arange(len(sig)) / fs
        ax.plot(t, sig, lw=0.8)
        ax.set_title(
            f"{_class_name(row['label'])} | folder {row['folder_id']} | {row['split']} | {row['subset']} | idx {row['sample_idx']}"
        )
        ax.set_ylabel("acc")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "time_domain_examples.png")
    plt.close(fig)


def plot_feature_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    plot_features = ["rms", "kurtosis", "skewness", "envelope_kurtosis", "E_bpfi", "E_bpfo", "E_bsf"]
    display_df = df.copy()
    display_df["class_name"] = display_df["label"].map(_class_name)

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    axes = axes.flatten()
    for i, feat in enumerate(plot_features):
        sns.histplot(data=display_df, x=feat, hue="class_name", kde=True, bins=40, ax=axes[i])
        axes[i].set_title(f"Histogram: {feat}")
    for j in range(len(plot_features), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_histograms.png")
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    axes = axes.flatten()
    for i, feat in enumerate(plot_features):
        sns.boxplot(data=display_df, x="class_name", y=feat, ax=axes[i])
        axes[i].set_title(f"Boxplot: {feat}")
    for j in range(len(plot_features), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_boxplots.png")
    plt.close(fig)


def plot_fft_and_envelope_examples(df: pd.DataFrame, output_dir: Path) -> None:
    chosen = pd.concat([df[df["label"] == 0].head(2), df[df["label"] == -1].head(2)])
    rows = chosen.to_dict("records")

    fig_fft, axes_fft = plt.subplots(len(rows), 1, figsize=(12, 2.7 * len(rows)), sharex=False)
    fig_env, axes_env = plt.subplots(len(rows), 1, figsize=(12, 2.7 * len(rows)), sharex=False)
    fig_envspec, axes_envspec = plt.subplots(len(rows), 1, figsize=(12, 2.7 * len(rows)), sharex=False)

    if len(rows) == 1:
        axes_fft = [axes_fft]
        axes_env = [axes_env]
        axes_envspec = [axes_envspec]

    for i, row in enumerate(rows):
        mat = loadmat(row["file_path"], squeeze_me=True, struct_as_record=False)
        struct = mat[row["subset"]]
        sig = np.asarray(struct.rawData[row["sample_idx"]], dtype=np.float64)
        fs = float(np.asarray(struct.samplingRate).reshape(-1)[row["sample_idx"]])
        rpm = float(np.asarray(struct.RPM).reshape(-1)[row["sample_idx"]])
        shaft_hz = rpm / 60.0 if rpm > 0 else np.nan

        # FFT
        spec = np.abs(np.fft.rfft(sig - np.mean(sig)))
        freq = np.fft.rfftfreq(len(sig), d=1.0 / fs)
        axes_fft[i].plot(freq, spec, lw=0.8)
        axes_fft[i].set_xlim(0, min(3000, fs / 2))
        axes_fft[i].set_title(f"FFT | {_class_name(row['label'])} | folder {row['folder_id']} {row['split']} {row['subset']}")

        # Overlay fault frequencies on FFT.
        for name, mult in [
            ("BPFI", row["fault_bpfi_mult"]),
            ("BPFO", row["fault_bpfo_mult"]),
            ("BSF", row["fault_bsf_mult"]),
            ("FTF", row["fault_ftf_mult"]),
        ]:
            if np.isfinite(shaft_hz):
                f0 = mult * shaft_hz
                axes_fft[i].axvline(f0, color="r", ls="--", alpha=0.4)
                axes_fft[i].text(f0, axes_fft[i].get_ylim()[1] * 0.8, name, rotation=90, fontsize=7)

        # Envelope signal
        env = _envelope(sig)
        t = np.arange(len(env)) / fs
        axes_env[i].plot(t, env, lw=0.8)
        axes_env[i].set_title(f"Envelope Signal | {_class_name(row['label'])}")
        axes_env[i].set_xlim(0, min(0.5, t[-1]))

        # Envelope spectrum
        ef, es = _envelope_spectrum(sig, fs)
        axes_envspec[i].plot(ef, es, lw=0.8)
        axes_envspec[i].set_xlim(0, min(800, fs / 2))
        axes_envspec[i].set_title(f"Envelope Spectrum | {_class_name(row['label'])}")

        for name, mult in [
            ("BPFI", row["fault_bpfi_mult"]),
            ("BPFO", row["fault_bpfo_mult"]),
            ("BSF", row["fault_bsf_mult"]),
            ("FTF", row["fault_ftf_mult"]),
        ]:
            if np.isfinite(shaft_hz):
                f0 = mult * shaft_hz
                axes_envspec[i].axvline(f0, color="r", ls="--", alpha=0.4)
                axes_envspec[i].text(f0, axes_envspec[i].get_ylim()[1] * 0.8, name, rotation=90, fontsize=7)

    axes_fft[-1].set_xlabel("Frequency (Hz)")
    axes_env[-1].set_xlabel("Time (s)")
    axes_envspec[-1].set_xlabel("Frequency (Hz)")
    fig_fft.tight_layout()
    fig_env.tight_layout()
    fig_envspec.tight_layout()

    fig_fft.savefig(output_dir / "fft_examples_with_fault_lines.png")
    fig_env.savefig(output_dir / "envelope_examples.png")
    fig_envspec.savefig(output_dir / "envelope_spectrum_examples.png")

    plt.close(fig_fft)
    plt.close(fig_env)
    plt.close(fig_envspec)


def window_level_stability(df: pd.DataFrame, output_dir: Path, window_size: int = 2048) -> pd.DataFrame:
    picked = pd.concat([df[df["label"] == 0].head(3), df[df["label"] == -1].head(3)])
    rows = []

    for _, row in picked.iterrows():
        mat = loadmat(row["file_path"], squeeze_me=True, struct_as_record=False)
        struct = mat[row["subset"]]
        sig = np.asarray(struct.rawData[int(row["sample_idx"])], dtype=np.float64)
        fs = float(np.asarray(struct.samplingRate).reshape(-1)[int(row["sample_idx"])])

        n_win = len(sig) // window_size
        for w in range(n_win):
            seg = sig[w * window_size : (w + 1) * window_size]
            env = _envelope(seg)
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "label": row["label"],
                    "window_idx": w,
                    "rms": float(np.sqrt(np.mean(seg ** 2))),
                    "kurtosis": float(kurtosis(seg, fisher=False, bias=False)),
                    "envelope_kurtosis": float(kurtosis(env, fisher=False, bias=False)),
                    "fs": fs,
                }
            )

    wdf = pd.DataFrame(rows)
    wdf["class_name"] = wdf["label"].map(_class_name)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    sns.lineplot(data=wdf, x="window_idx", y="rms", hue="sample_id", style="class_name", marker="o", ax=axes[0])
    axes[0].set_title("Window-Level RMS Stability")
    sns.lineplot(
        data=wdf,
        x="window_idx",
        y="kurtosis",
        hue="sample_id",
        style="class_name",
        marker="o",
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("Window-Level Kurtosis Stability")
    sns.lineplot(
        data=wdf,
        x="window_idx",
        y="envelope_kurtosis",
        hue="sample_id",
        style="class_name",
        marker="o",
        ax=axes[2],
        legend=False,
    )
    axes[2].set_title("Window-Level Envelope Kurtosis Stability")
    axes[2].set_xlabel("Window Index")
    fig.tight_layout()
    fig.savefig(output_dir / "window_level_feature_stability.png")
    plt.close(fig)

    stability = (
        wdf.groupby(["sample_id", "class_name"])[["rms", "kurtosis", "envelope_kurtosis"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    if isinstance(stability.columns, pd.MultiIndex):
        stability.columns = [
            "_".join([str(c) for c in col if str(c) != ""]).strip("_") for col in stability.columns.to_flat_index()
        ]
    return stability


def leakage_and_bias_checks(df: pd.DataFrame) -> Dict[str, float]:
    bin_df = df[df["label"].isin([0, -1])].copy()
    x = bin_df[["rpm"]].values
    y = (bin_df["label"] == -1).astype(int).values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    aucs = []
    for tr, te in skf.split(x, y):
        clf = LogisticRegression(max_iter=200)
        clf.fit(x[tr], y[tr])
        p = clf.predict_proba(x[te])[:, 1]
        yhat = (p >= 0.5).astype(int)
        accs.append(accuracy_score(y[te], yhat))
        aucs.append(roc_auc_score(y[te], p))

    # Grouped CV by folder+split+subset to reduce file-level leakage effect.
    groups = bin_df["folder_id"] + "_" + bin_df["split"] + "_" + bin_df["subset"]
    gkf = GroupKFold(n_splits=5)
    g_accs = []
    g_aucs = []
    for tr, te in gkf.split(x, y, groups=groups):
        clf = LogisticRegression(max_iter=200)
        clf.fit(x[tr], y[tr])
        p = clf.predict_proba(x[te])[:, 1]
        yhat = (p >= 0.5).astype(int)
        g_accs.append(accuracy_score(y[te], yhat))
        g_aucs.append(roc_auc_score(y[te], p))

    # Exact duplicate hashes across train/test indicate potential overlap risk.
    split_hashes = df.groupby("split")["signal_hash"].apply(set).to_dict()
    overlap = 0
    if "train" in split_hashes and "test" in split_hashes:
        overlap = len(split_hashes["train"].intersection(split_hashes["test"]))

    return {
        "binary_samples_used": int(len(bin_df)),
        "rpm_only_cv_acc_mean": float(np.mean(accs)),
        "rpm_only_cv_auc_mean": float(np.mean(aucs)),
        "rpm_only_groupcv_acc_mean": float(np.mean(g_accs)),
        "rpm_only_groupcv_auc_mean": float(np.mean(g_aucs)),
        "train_test_exact_duplicate_signals": int(overlap),
    }


def quality_checks(df: pd.DataFrame) -> Dict[str, float]:
    q = {}
    q["total_samples"] = int(len(df))
    q["missing_rpm"] = int(df["rpm"].isna().sum())
    q["zero_rpm"] = int((df["rpm"] == 0).sum())
    q["nan_signal_values"] = int(df["nan_count"].sum())
    q["inf_signal_values"] = int(df["inf_count"].sum())
    q["non_binary_labels"] = int((~df["label"].isin([0, -1])).sum())
    q["outliers_rms_3sigma"] = int(((np.abs((df["rms"] - df["rms"].mean()) / df["rms"].std()) > 3)).sum())
    q["outliers_kurtosis_3sigma"] = int(
        ((np.abs((df["kurtosis"] - df["kurtosis"].mean()) / df["kurtosis"].std()) > 3)).sum()
    )
    return q


def write_report(
    output_dir: Path,
    records_df: pd.DataFrame,
    files_df: pd.DataFrame,
    window_stability_df: pd.DataFrame,
    quality: Dict[str, float],
    leakage: Dict[str, float],
) -> None:
    total_files = files_df["file_path"].nunique()
    total_samples = len(records_df)
    healthy = int((records_df["label"] == 0).sum())
    faulty = int((records_df["label"] == -1).sum())
    non_binary = int((~records_df["label"].isin([0, -1])).sum())
    imbalance_ratio = max(healthy, faulty) / max(1, min(healthy, faulty))

    ds_stats = records_df[records_df["subset"] == "DS"]
    fs_stats = records_df[records_df["subset"] == "FS"]

    feat_cols = ["E_bpfi", "E_bpfo", "E_bsf", "kurtosis", "envelope_kurtosis", "rms", "skewness"]
    feat_summary = (
        records_df.assign(class_name=records_df["label"].map(_class_name))
        .groupby("class_name")[feat_cols]
        .agg(["mean", "std", "median"])
        .round(5)
    )
    feat_summary.to_csv(output_dir / "feature_summary_by_class.csv")

    # Per-file imbalance summary.
    per_file = (
        records_df.groupby(["folder_id", "split", "subset", "label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={-1: "faulty", 0: "healthy"})
        .reset_index()
    )
    per_file["imbalance_ratio_major_minor"] = per_file[["healthy", "faulty"]].max(axis=1) / per_file[
        ["healthy", "faulty"]
    ].min(axis=1).replace(0, np.nan)
    per_file.to_csv(output_dir / "label_distribution_per_file.csv", index=False)

    structure_summary = {
        "total_mat_files": int(total_files),
        "total_samples": int(total_samples),
        "folders": int(records_df["folder_id"].nunique()),
        "signal_length_unique": sorted(records_df["signal_length"].unique().tolist()),
        "sampling_rate_unique": sorted(records_df["sampling_rate"].unique().tolist()),
    }
    with open(output_dir / "structure_summary.json", "w", encoding="utf-8") as f:
        json.dump(structure_summary, f, indent=2)

    with open(output_dir / "leakage_bias_summary.json", "w", encoding="utf-8") as f:
        json.dump(leakage, f, indent=2)

    with open(output_dir / "quality_summary.json", "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)

    # Text report with section-level interpretation.
    report = []
    report.append("# SCA Bearing Dataset: Deep Programmatic Study\n")
    report.append("This report is generated from code in `sca_dataset_study.py`.\n")
    report.append("\n## 1) Dataset Structure Analysis\n")
    report.append(f"- Total files: {total_files}\n")
    report.append(f"- Total samples across DS+FS: {total_samples}\n")
    report.append(f"- Folders: {records_df['folder_id'].nunique()}\n")
    report.append(f"- Unique signal lengths: {sorted(records_df['signal_length'].unique().tolist())}\n")
    report.append(f"- Unique sampling rates: {sorted(records_df['sampling_rate'].unique().tolist())}\n")
    report.append(
        f"- DS samples: {len(ds_stats)} | FS samples: {len(fs_stats)}\n"
    )
    report.append(
        "Interpretation: DS and FS have similar schema and are both usable for feature extraction; verify domain shift by comparing feature distributions across subsets before final model training.\n"
    )

    report.append("\n## 2) Label Distribution\n")
    report.append(f"- Healthy: {healthy}\n")
    report.append(f"- Faulty: {faulty}\n")
    report.append(f"- Non-binary labels (unexpected): {non_binary}\n")
    report.append(f"- Imbalance ratio (major/minor): {imbalance_ratio:.3f}\n")
    report.append(
        "Interpretation: If ratio is materially above 1.5, class weighting and threshold calibration are recommended; if near 1.0, simpler balanced objectives are acceptable.\n"
    )
    report.append("- Plots: `label_distribution_overall.png`, `label_distribution_per_file.png`\n")

    report.append("\n## 3) RPM Analysis\n")
    report.append(
        f"- RPM mean healthy: {records_df.loc[records_df['label']==0, 'rpm'].mean():.3f} | faulty: {records_df.loc[records_df['label']==-1, 'rpm'].mean():.3f}\n"
    )
    report.append(
        "Interpretation: Large RPM separation between classes can cause shortcut learning. Use grouped splits and/or RPM normalization/order tracking.\n"
    )
    report.append("- Plots: `rpm_distribution_by_label.png`, `rpm_vs_label_boxplot.png`\n")

    report.append("\n## 4) Signal Analysis (Time Domain)\n")
    report.append("- Computed per sample: RMS, kurtosis, skewness\n")
    report.append("- Plots: `time_domain_examples.png`, `feature_histograms.png`, `feature_boxplots.png`\n")
    report.append(
        "Interpretation: RMS captures global energy, while kurtosis/envelope-kurtosis are more sensitive to impulsive defects.\n"
    )

    report.append("\n## 5) Frequency Domain Analysis\n")
    report.append("- FFT generated for healthy/faulty samples with fault-frequency overlays.\n")
    report.append("- Plot: `fft_examples_with_fault_lines.png`\n")
    report.append(
        "Interpretation: Spectral peaks near characteristic fault bands provide physically meaningful evidence of bearing defects.\n"
    )

    report.append("\n## 6) Envelope Analysis\n")
    report.append("- Envelope via Hilbert transform and envelope spectrum extracted.\n")
    report.append("- Plots: `envelope_examples.png`, `envelope_spectrum_examples.png`\n")
    report.append(
        "Interpretation: Envelope spectrum tends to enhance amplitude modulation components and is often more discriminative than raw FFT for localized defects.\n"
    )

    report.append("\n## 7) Feature Distribution Study\n")
    report.append("- Features: E_bpfi, E_bpfo, E_bsf, kurtosis, envelope_kurtosis (plus RMS/skewness).\n")
    report.append("- Saved table: `feature_summary_by_class.csv`\n")
    report.append(
        "Interpretation: Features with higher class-wise mean shift and lower overlap in boxplots are stronger candidates for robust classification.\n"
    )

    report.append("\n## 8) Window-Level Analysis\n")
    report.append("- Window size: 2048 points\n")
    report.append("- Plot: `window_level_feature_stability.png`\n")
    report.append("- Table: `window_stability_summary.csv`\n")
    report.append(
        "Interpretation: High within-sample variance implies unstable fault signatures and supports window aggregation (median/percentile pooling).\n"
    )

    report.append("\n## 9) Data Quality Issues\n")
    for k, v in quality.items():
        report.append(f"- {k}: {v}\n")
    report.append(
        "Interpretation: Non-zero missing/invalid signal values require cleaning; zero RPM samples should be flagged because fault-order mapping depends on RPM.\n"
    )

    report.append("\n## 10) Leakage & Bias Check\n")
    for k, v in leakage.items():
        report.append(f"- {k}: {v}\n")
    report.append(
        "Interpretation: If RPM-only metrics are high, model may rely on operating conditions rather than fault physics. Grouped CV provides a safer estimate.\n"
    )

    report.append("\n## 11) Research Insights\n")
    report.append(
        "- Most reliable features generally are envelope-kurtosis and fault-band energies when they remain stable across folders and speeds.\n"
    )
    report.append(
        "- Misleading features are those strongly correlated with RPM or file identity rather than defect impulsiveness.\n"
    )
    report.append(
        "- The hardest challenge is separating true fault evidence from operating-condition variation (speed/load domain shift).\n"
    )

    report.append("\n## 12) Recommendations\n")
    report.append(
        "- Use classification with grouped splits first; use one-class/novelty setup if faulty labels are scarce or unreliable in deployment.\n"
    )
    report.append(
        "- Preprocess: detrend -> bandpass (optional) -> Hilbert envelope -> order-aware features using RPM.\n"
    )
    report.append(
        "- Candidate feature set: [RMS, kurtosis, envelope_kurtosis, E_bpfi, E_bpfo, E_bsf] + robust window pooling stats.\n"
    )

    report_path = output_dir / "SCA_dataset_study_report.md"
    report_path.write_text("".join(report), encoding="utf-8")


def main() -> None:
    root = Path("/home/teaching/Hackathon_dl")
    dataset_root = root / "SCA bearing dataset"
    output_dir = root / "analysis_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    records_df, files_df = collect_dataset_records(dataset_root)
    records_df.to_csv(output_dir / "sample_level_features.csv", index=False)
    files_df.to_csv(output_dir / "file_level_summary.csv", index=False)

    plot_label_distribution(records_df, output_dir)
    plot_rpm_analysis(records_df, output_dir)
    plot_time_domain_examples(dataset_root, records_df, output_dir)
    plot_feature_distributions(records_df, output_dir)
    plot_fft_and_envelope_examples(records_df, output_dir)
    window_stability_df = window_level_stability(records_df, output_dir)
    window_stability_df.to_csv(output_dir / "window_stability_summary.csv", index=False)

    quality = quality_checks(records_df)
    leakage = leakage_and_bias_checks(records_df)

    write_report(output_dir, records_df, files_df, window_stability_df, quality, leakage)

    print("Analysis complete.")
    print(f"Output directory: {output_dir}")
    print(f"Total samples: {len(records_df)}")
    print(f"Healthy samples: {(records_df['label'] == 0).sum()}")
    print(f"Faulty samples: {(records_df['label'] == -1).sum()}")
    print("Saved key files:")
    for name in [
        "SCA_dataset_study_report.md",
        "sample_level_features.csv",
        "file_level_summary.csv",
        "feature_summary_by_class.csv",
        "label_distribution_overall.png",
        "rpm_distribution_by_label.png",
        "fft_examples_with_fault_lines.png",
        "envelope_spectrum_examples.png",
    ]:
        print("-", output_dir / name)


if __name__ == "__main__":
    main()
