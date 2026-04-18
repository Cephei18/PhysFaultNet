# SCA Bearing Dataset: Deep Programmatic Study
This report is generated from code in `sca_dataset_study.py`.

## 1) Dataset Structure Analysis
- Total files: 22
- Total samples across DS+FS: 6644
- Folders: 11
- Unique signal lengths: [8192, 16384]
- Unique sampling rates: [512.0, 640.0, 2560.0, 4096.0, 5120.0, 6400.0, 8192.0, 12800.0]
- DS samples: 2979 | FS samples: 2870
Interpretation: DS and FS have similar schema and are both usable for feature extraction; verify domain shift by comparing feature distributions across subsets before final model training.

## 2) Label Distribution
- Healthy: 5598
- Faulty: 198
- Non-binary labels (unexpected): 848
- Imbalance ratio (major/minor): 28.273
Interpretation: If ratio is materially above 1.5, class weighting and threshold calibration are recommended; if near 1.0, simpler balanced objectives are acceptable.
- Plots: `label_distribution_overall.png`, `label_distribution_per_file.png`

## 3) RPM Analysis
- RPM mean healthy: 1036.301 | faulty: 222.296
Interpretation: Large RPM separation between classes can cause shortcut learning. Use grouped splits and/or RPM normalization/order tracking.
- Plots: `rpm_distribution_by_label.png`, `rpm_vs_label_boxplot.png`

## 4) Signal Analysis (Time Domain)
- Computed per sample: RMS, kurtosis, skewness
- Plots: `time_domain_examples.png`, `feature_histograms.png`, `feature_boxplots.png`
Interpretation: RMS captures global energy, while kurtosis/envelope-kurtosis are more sensitive to impulsive defects.

## 5) Frequency Domain Analysis
- FFT generated for healthy/faulty samples with fault-frequency overlays.
- Plot: `fft_examples_with_fault_lines.png`
Interpretation: Spectral peaks near characteristic fault bands provide physically meaningful evidence of bearing defects.

## 6) Envelope Analysis
- Envelope via Hilbert transform and envelope spectrum extracted.
- Plots: `envelope_examples.png`, `envelope_spectrum_examples.png`
Interpretation: Envelope spectrum tends to enhance amplitude modulation components and is often more discriminative than raw FFT for localized defects.

## 7) Feature Distribution Study
- Features: E_bpfi, E_bpfo, E_bsf, kurtosis, envelope_kurtosis (plus RMS/skewness).
- Saved table: `feature_summary_by_class.csv`
Interpretation: Features with higher class-wise mean shift and lower overlap in boxplots are stronger candidates for robust classification.

## 8) Window-Level Analysis
- Window size: 2048 points
- Plot: `window_level_feature_stability.png`
- Table: `window_stability_summary.csv`
Interpretation: High within-sample variance implies unstable fault signatures and supports window aggregation (median/percentile pooling).

## 9) Data Quality Issues
- total_samples: 6644
- missing_rpm: 0
- zero_rpm: 154
- nan_signal_values: 0
- inf_signal_values: 0
- non_binary_labels: 848
- outliers_rms_3sigma: 156
- outliers_kurtosis_3sigma: 8
Interpretation: Non-zero missing/invalid signal values require cleaning; zero RPM samples should be flagged because fault-order mapping depends on RPM.

## 10) Leakage & Bias Check
- binary_samples_used: 5796
- rpm_only_cv_acc_mean: 0.9658385647555858
- rpm_only_cv_auc_mean: 0.8838660668252659
- rpm_only_groupcv_acc_mean: 0.965718199133202
- rpm_only_groupcv_auc_mean: 0.940211488879263
- train_test_exact_duplicate_signals: 0
Interpretation: If RPM-only metrics are high, model may rely on operating conditions rather than fault physics. Grouped CV provides a safer estimate.

## 11) Research Insights
- Most reliable features generally are envelope-kurtosis and fault-band energies when they remain stable across folders and speeds.
- Misleading features are those strongly correlated with RPM or file identity rather than defect impulsiveness.
- The hardest challenge is separating true fault evidence from operating-condition variation (speed/load domain shift).

## 12) Recommendations
- Use classification with grouped splits first; use one-class/novelty setup if faulty labels are scarce or unreliable in deployment.
- Preprocess: detrend -> bandpass (optional) -> Hilbert envelope -> order-aware features using RPM.
- Candidate feature set: [RMS, kurtosis, envelope_kurtosis, E_bpfi, E_bpfo, E_bsf] + robust window pooling stats.
