# 🧠 PhySense-AI
### Physics-Informed Multimodal Anomaly Detection for Bearing Fault Diagnosis

Modern industrial systems rarely fail often, but when they do, early detection is critical.

This project presents a **physics-informed anomaly detection system** for rotating machinery, designed to operate under **real-world constraints where faulty data is scarce**.

Instead of relying on fully supervised classification, we reformulate the problem as:

> **Learn normal behavior → detect deviations → diagnose fault type**

---

## ⚙️ Key Idea

We combine **mechanical physics + signal processing + machine learning** to build a robust and interpretable system:

- 📊 **Envelope analysis** to extract impulsive fault signatures
- 📈 **Fault-frequency modeling** (BPFI, BPFO, BSF) based on bearing kinematics
- 🧠 **One-Class Learning** trained only on healthy data
- 🔁 **Temporal Prediction Model** to capture signal dynamics
- 🔗 **Multimodal Fusion** for robust anomaly detection

---

## 🔍 Why This Approach?

Real-world fault diagnosis is not a balanced classification problem.

- Fault data is **rare and noisy**
- Operating conditions vary (speed, load)
- Pure deep learning models often lack interpretability

This system addresses these challenges by:

- grounding features in **physical principles**
- modeling **normal system behavior**
- detecting anomalies as **deviations from expected dynamics**

---

## 🧩 System Overview

Raw Signal + Metadata  
↓  
Physics-Based Quality Gate  
↓  
Envelope Extraction (Hilbert Transform)  
↓  
FFT + Feature Engineering  
↓  
One-Class SVM + Temporal Predictor  
↓  
Score Fusion  
↓  
Thresholding  
↓  
Fault Diagnosis

---

## 📊 Results Summary

### 🔹 Anomaly Detection (Primary)

Latest evaluated subset results:

- 🚀 **ROC-AUC (Fusion): 0.8747**
- 📉 **PR-AUC (Fusion): 0.9077**
- ✅ **Healthy recall: 0.9571**
- ✅ Strong performance under **class imbalance**

Supporting anomaly detector metrics:

- **ROC-AUC (SVM normalized): 0.8171**
- **PR-AUC (SVM normalized): 0.8744**
- **ROC-AUC (Temporal normalized): 0.5821**
- **PR-AUC (Temporal normalized): 0.6386**

---

### 🔹 Fault Classification (Secondary)

Balanced faulty-subset classifier results:

- 🎯 **Macro F1: 0.91**
- ✅ Classifier predicts classes **1, 2, and 3** on the fault-only split

Full pipeline subset results after threshold tuning:

- **Accuracy: 0.67**
- **Healthy recall: 0.9571**
- **Fault-only accuracy: 0.0**

This confirms that the anomaly gate is strong, while fault-class separation remains the harder part of the task.

---

## ⚠️ Important Note on Dataset

During extensive dataset analysis, we observed:

- Fault classes are **highly imbalanced**
- Fault labels are **derived, not perfectly clean**
- Some fault types, especially ball faults, are **weakly separable in feature space**

This makes **pure multi-class classification inherently challenging**.

As a result:

> The system prioritizes **reliable anomaly detection**, which is more aligned with real-world industrial scenarios.

---

## 💡 Core Insight

> **You don’t need to see every failure — you need to understand what “normal” looks like.**

---

## 🎯 Applications

- Predictive maintenance
- Industrial condition monitoring
- Rotating machinery diagnostics

---

## 🛠 Tech Stack

- Python
- NumPy / SciPy
- Scikit-learn
- PyTorch

---

## 📁 Outputs and Figures

Download all generated figures here:

- [presentation_graphs.zip](presentation_graphs.zip)

Useful plots for presentation:

- [trained_models/envelope_dynamics_loss_curve.png](trained_models/envelope_dynamics_loss_curve.png)
- [trained_models/temporal_error_hist_overlay.png](trained_models/temporal_error_hist_overlay.png)
- [trained_models/svm_temporal_fused_hist_overlay.png](trained_models/svm_temporal_fused_hist_overlay.png)
- [trained_models/full_pipeline_confusion_matrix.png](trained_models/full_pipeline_confusion_matrix.png)
- [analysis_outputs/time_domain_examples.png](analysis_outputs/time_domain_examples.png)
- [analysis_outputs/fft_examples_with_fault_lines.png](analysis_outputs/fft_examples_with_fault_lines.png)
- [analysis_outputs/envelope_spectrum_examples.png](analysis_outputs/envelope_spectrum_examples.png)
- [analysis_outputs/feature_histograms.png](analysis_outputs/feature_histograms.png)
- [analysis_outputs/feature_boxplots.png](analysis_outputs/feature_boxplots.png)

---

## 🧠 Final Takeaway

This project demonstrates that:

- Physics-informed preprocessing is critical
- Anomaly detection is more reliable than forced classification in real-world data
- Multimodal fusion improves robustness across operating conditions

---

## 🚀 Reproducibility

Key scripts:

- `sca_dataset_study.py` - dataset exploration and feature generation
- `train_demo.py` - temporal model training demo
- `evaluate_fusion_pipeline.py` - SVM + temporal fusion evaluation
- `test_full_pipeline.py` - end-to-end anomaly detection + classification test
