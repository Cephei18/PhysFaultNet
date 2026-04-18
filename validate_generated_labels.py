"""Validate multiclass labels generated from fault-energy features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models import generate_multiclass_labels


def main() -> None:
    root = Path("/home/teaching/Hackathon_dl")
    csv_path = root / "analysis_outputs" / "sample_level_features.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input file: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["label", "E_bpfi", "E_bpfo", "E_bsf"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Build a minimal feature matrix where indices 2/3/4 map to E_bpfi/E_bpfo/E_bsf.
    features = np.c_[
        np.zeros(len(df), dtype=np.float64),
        np.zeros(len(df), dtype=np.float64),
        df["E_bpfi"].to_numpy(dtype=np.float64),
        df["E_bpfo"].to_numpy(dtype=np.float64),
        df["E_bsf"].to_numpy(dtype=np.float64),
    ]
    binary_labels = df["label"].to_numpy()

    y_multiclass = generate_multiclass_labels(features, binary_labels)

    # 1) unique values
    unique_vals = np.unique(y_multiclass)
    print("Unique values in y_multiclass:", unique_vals)

    # 2) class counts for 0,1,2,3
    print("Class counts:")
    for cls in [0, 1, 2, 3]:
        cnt = int(np.sum(y_multiclass == cls))
        print(f"  class {cls}: {cnt}")

    # Ensure all 4 classes exist if possible
    existing = set(int(v) for v in unique_vals)
    target = {0, 1, 2, 3}
    missing_classes = sorted(target - existing)
    if len(missing_classes) == 0:
        print("All 4 classes exist: True")
    else:
        print("All 4 classes exist: False")
        print("Missing classes:", missing_classes)

    # 3) 10 faulty samples with energies and assigned label
    faulty_idx = np.where(binary_labels != 0)[0]
    n_show = min(10, len(faulty_idx))
    print(f"\nFaulty sample preview ({n_show} rows):")

    match_flags: list[bool] = []
    for i in faulty_idx[:n_show]:
        e_bpfi = float(features[i, 2])
        e_bpfo = float(features[i, 3])
        e_bsf = float(features[i, 4])
        assigned = int(y_multiclass[i])

        energies = np.array([e_bpfi, e_bsf, e_bpfo], dtype=np.float64)
        idx = int(np.argmax(energies))
        expected = {0: 1, 1: 2, 2: 3}[idx]
        ok = assigned == expected
        match_flags.append(ok)

        print(
            f"  E_bpfi={e_bpfi:.6f}, E_bpfo={e_bpfo:.6f}, E_bsf={e_bsf:.6f}, "
            f"assigned={assigned}, match_highest={ok}"
        )

    # 4) consistency check over all faulty samples
    if len(faulty_idx) > 0:
        faulty_feats = features[faulty_idx]
        idx_all = np.argmax(np.c_[faulty_feats[:, 2], faulty_feats[:, 4], faulty_feats[:, 3]], axis=1)
        expected_all = np.array([1 if t == 0 else 2 if t == 1 else 3 for t in idx_all], dtype=np.int32)
        assigned_all = y_multiclass[faulty_idx]
        overall_match = bool(np.all(expected_all == assigned_all))
        print(f"\nAssigned label matches highest energy (all faulty): {overall_match}")
    else:
        print("\nAssigned label matches highest energy (all faulty): N/A (no faulty samples)")


if __name__ == "__main__":
    main()
