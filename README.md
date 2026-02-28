# Heartbeat or Noise?
### Evaluating Consumer Wearables as Cardiac Screening Proxies Using Machine Learning

**Author:** Lucas Sam
**Program:** General Assembly Data Analytics Capstone Project
**Date:** February 2026

---

## Overview

Cardiovascular disease accounts for 30.5% of all deaths in Singapore, equating to approximately 22 lives lost daily. Despite national screening efforts, approximately 49% of Singapore residents did not undergo a health checkup in the past twelve months, leaving a significant proportion of the population unaware of underlying cardiac risk.

Consumer wearables — including smartwatches from Apple and Garmin — are worn by millions daily and continuously monitor physiological signals including heart rate and heart rate variability. This project evaluates whether machine learning models trained on clinical ECG data can generalise to consumer wearable signals to detect cardiac rhythm irregularities, assessing their feasibility as low-barrier proxy screening tools that could prompt at-risk individuals toward formal clinical evaluation.

---

## Research Question

Can machine learning models trained on clinical cardiac data generalise to consumer wearable signals to detect heart rhythm irregularities, and can consumer wearables feasibly serve as proxy screening tools to identify individuals who would benefit from formal cardiac evaluation?

---

## Project Contribution

While Apple's AF detection algorithm represents a clinically validated application of consumer wearable cardiac monitoring, its scope is limited to atrial fibrillation specifically and its methodology remains proprietary and non-reproducible. This project extends beyond AF detection to classify a broader range of cardiac rhythm irregularities — normal versus any abnormal rhythm warranting clinical attention — using a transparent, reproducible machine learning methodology built entirely on publicly available clinical data. In doing so, it provides an open, auditable framework for evaluating whether consumer wearables can serve as a low-barrier proxy screening tool capable of prompting at-risk individuals toward formal clinical evaluation before an acute cardiac event occurs.

---

## Datasets

### Primary — Physionet 2017 AF Classification Challenge
- 8,528 single-lead ECG recordings sampled at 300 Hz
- 8,249 usable recordings after excluding 279 noisy-class (`~`) records
- Binary label mapping: Normal (N → 0) versus Abnormal (AF + Other → 1)
- Class distribution: 5,076 Normal, 758 AF, 2,415 Other
- Source: https://physionet.org/content/challenge-2017/1.0.0/

### Secondary — Personal Apple Watch Data
- **Device:** Apple Watch SE 1st Generation (Model A2352)
- **Source:** 1.58 GB Apple Health XML export (3.79M total records, 515K from Watch)
- **Temporal coverage:** April 2021 — February 2026 (nearly 5 years)
- **Extracted metrics:**

| Metric | Records | File |
|--------|--------:|------|
| Heart rate (continuous) | 478,078 | `heart_rate_raw.csv` |
| Respiratory rate | 20,372 | `respiratory_rate_raw.csv` |
| Sleep analysis | 7,967 | `sleep_raw.csv` |
| Heart rate variability (SDNN) | 5,485 | `hrv_raw.csv` |
| Workouts | 1,982 | `workouts_raw.csv` |
| Walking heart rate average | 1,557 | `walking_hr_raw.csv` |
| Resting heart rate | 1,491 | `resting_hr_raw.csv` |
| VO2 Max | 126 | `vo2max_raw.csv` |
| Heart rate recovery (1 min) | 49 | `hr_recovery_raw.csv` |

- **Clinical anchor point:** ECG conducted June 18, 2025 confirming cardiac irregularity
- **Role:** Consumer-grade signal bridge layer — not used for model training or validation

---

## Methodology

### Layer 1 — Clinical Benchmark
A binary cardiac rhythm classifier is trained and validated on the Physionet dataset using only features extractable from consumer wearable devices. This deliberate constraint ensures the model reflects real-world deployment conditions. Sensitivity to abnormal rhythms is the primary evaluation metric, consistent with the screening context in which missing a true positive carries greater consequence than generating a false alarm.

**Minimum performance threshold:** Sensitivity ≥ 0.80 at Specificity ≥ 0.75

**Locked feature set** (8 features, frequency domain excluded):
| Feature | Description |
|---------|-------------|
| RMSSD | Root mean square of successive RR interval differences |
| SDNN | Standard deviation of all RR intervals |
| Mean RR | Average inter-beat interval (ms) |
| pNN50 | Proportion of successive intervals differing by >50ms |
| HR Mean | Average heart rate derived from RR intervals |
| HR Std Dev | Variability of instantaneous heart rate |
| RR Skewness | Asymmetry of the RR interval distribution |
| RR Kurtosis | Tail weight of the RR interval distribution |

### Layer 2 — Consumer Wearable Bridge
The trained model is applied to personal Apple Watch data to evaluate signal transferability from clinical ECG to consumer PPG-derived measurements. Success is assessed across three tiers:

- **Tier 1 — Foundational Applicability:** The feature pipeline executes on Apple Watch data with interpretable outputs and documented signal quality differences between clinical and consumer measurements.
- **Tier 2 — Signal Detection:** Model output scores show statistically meaningful deviation from personal baseline during the ninety-day window surrounding the June 2025 clinical ECG event.
- **Tier 3 — Clinical Concordance:** The ECG report confirms an irregularity consistent with the model's detection scope and the Apple Watch data demonstrates the Tier 2 deviation pattern.

---

## Current Progress

### Completed

**Phase 1 — Data Acquisition and Exploration** (`01_data_exploration.ipynb`)
- Physionet 2017 dataset validated: 8,528 `.mat` + `.hea` file pairs, class distribution confirmed
- Apple Health XML fully parsed: 3.79M records from 8 source devices identified
- Apple Watch data extracted: 515,125 records across 9 metric types saved as raw CSVs
- Sample ECG waveform visualised and saved (`outputs/figures/sample_ecg.png`)

**Phase 2 — Data Preprocessing** (`02_feature_engineering.ipynb`)
- Apple Watch cleaning pipeline built (`src/preprocess.py`) and executed across 5 cardiac metrics
- Cleaning results: >99% data retention across all metrics (29 HRV records removed as non-numeric, all other metrics 100% retained)
- Post-pipeline artifact removal: 1 confirmed sensor artifact (210 bpm during sleep on 2021-08-28) removed from `heart_rate_clean.csv`
- Anchor period labels assigned: baseline (>91 days before ECG), pre_anchor (90 days before), post_anchor (90 days after), follow_up (>90 days after)
- Heart rate distribution validated: mean 84.8 bpm, median 82.0 bpm, range 39–205 bpm
- 7 remaining edge cases documented (2× sleep bradycardia at 39 bpm, 5× exercise tachycardia at 201–205 bpm) — retained as physiologically plausible

**Phase 3 — Feature Engineering** (`02_feature_engineering.ipynb` + `src/features.py`)
- Feature extraction pipeline built (`src/features.py`) for Physionet clinical ECG recordings
- 8 locked HRV features defined and constrained to wearable-extractable measurements
- R-peak detection via XQRS with quality filtering (RR intervals: 300–2000ms, minimum 10 valid intervals per recording)
- Binary label mapping applied: N → 0 (Normal), A/O → 1 (Abnormal), `~` excluded
- Feature extraction in progress across 8,249 recordings — output will be saved to `data/processed/physionet_features.csv`

### Remaining

- **Phase 4 — Model Development:** Train baseline Logistic Regression, Random Forest, XGBoost, and SVM classifiers
- **Phase 5 — Model Evaluation:** Evaluate sensitivity (primary), AUROC, F1, specificity; optimise classification threshold for screening
- **Phase 6 — Consumer Wearable Bridge:** Apply feature pipeline to Apple Watch data, quantify signal degradation, analyse pre/post ECG anchor patterns
- **Phase 7 — Conclusions:** Answer research question, document Singapore public health implications, limitations, and future work

---

## Project Structure
```
ga-capstone-heartbeat-or-noise/
│   README.md
│   requirements.txt
│   CLAUDE.md
│
├───data/
│   ├───physionet/            # 8,528 ECG recordings (.mat + .hea) + REFERENCE-v3.csv
│   ├───apple_watch/          # Raw CSV exports from Apple Health XML
│   │   ├───export.xml        # Source XML (1.58 GB, not committed)
│   │   ├───heart_rate_raw.csv
│   │   ├───hrv_raw.csv
│   │   ├───resting_hr_raw.csv
│   │   ├───walking_hr_raw.csv
│   │   ├───respiratory_rate_raw.csv
│   │   ├───sleep_raw.csv
│   │   ├───vo2max_raw.csv
│   │   ├───hr_recovery_raw.csv
│   │   └───workouts_raw.csv
│   └───processed/            # Cleaned and feature-engineered outputs
│       ├───heart_rate_clean.csv
│       ├───hrv_clean.csv
│       ├───resting_hr_clean.csv
│       ├───walking_hr_clean.csv
│       ├───respiratory_rate_clean.csv
│       └───physionet_features.csv  (pending — feature extraction in progress)
│
├───notebooks/
│   ├───01_data_exploration.ipynb       # Data acquisition, XML parsing, validation
│   ├───02_feature_engineering.ipynb    # Cleaning pipeline, artifact removal, feature extraction
│   ├───03_modelling.ipynb              # (planned)
│   ├───04_apple_watch_bridge.ipynb     # (planned)
│   └───05_conclusions.ipynb            # (planned)
│
├───src/
│   ├───preprocess.py     # Apple Watch data cleaning pipeline
│   ├───features.py       # Physionet ECG feature extraction (8 HRV features)
│   └───evaluate.py       # Model evaluation functions (planned)
│
└───outputs/
    ├───figures/           # sample_ecg.png
    └───models/            # (planned — saved model files)
```

---

## Notebook Guide

| Notebook | Purpose | Key Outputs |
|----------|---------|-------------|
| `01_data_exploration` | Validate Physionet dataset, parse Apple Health XML, extract raw CSVs | 9 raw CSV files in `data/apple_watch/`, `sample_ecg.png` |
| `02_feature_engineering` | Clean Apple Watch data, remove artifacts, extract Physionet features | 5 cleaned CSVs in `data/processed/`, `physionet_features.csv` |
| `03_modelling` | Train and compare binary classifiers | (planned) |
| `04_apple_watch_bridge` | Apply model to Apple Watch data, analyse anchor periods | (planned) |
| `05_conclusions` | Answer research question, document limitations | (planned) |

### Source Modules

| Module | Called By | Purpose |
|--------|-----------|---------|
| `src/preprocess.py` | `02_feature_engineering` | Cleans 5 Apple Watch metrics: date parsing, non-numeric removal, threshold filtering (30–220 bpm HR, 5–200ms HRV, etc.), anchor period assignment |
| `src/features.py` | `02_feature_engineering` | Extracts 8 HRV features from Physionet ECG recordings: R-peak detection (XQRS), RR interval computation, quality filtering, feature calculation |
| `src/evaluate.py` | `03_modelling` | (planned) |

---

## Environment Setup
```bash
# Clone the repository
git clone https://github.com/Lucas-sam93/ga-capstone-heartbeat-or-noise.git

# Create and activate conda environment
conda create -n cvd_project python=3.13
conda activate cvd_project

# Install dependencies
conda install numpy pandas scipy scikit-learn matplotlib seaborn jupyter ipykernel -y
pip install wfdb
```

---

## Key Methodological Decisions

| Decision | Choice | Justification |
|---|---|---|
| Primary dataset | Physionet 2017 | Clinically validated, single-lead ECG matches wearable modality |
| Noisy class | Excluded (279 records) | Signal quality failure, not rhythm classification |
| Classification type | Binary — Normal vs Abnormal | Screening requires triage decision, not diagnosis |
| Feature constraint | 8 wearable-extractable HRV features | Ensures valid generalisation test — model operates on signals wearables can produce |
| Frequency domain | Excluded | Decision made Feb 2026 — time domain features sufficient for initial benchmark |
| Primary metric | Sensitivity | Missing a true positive is more dangerous than a false positive in screening |
| Apple Watch role | Bridge layer only | N=1, no clinical outcome labels |
| Anchor date | June 18, 2025 | Date of clinical ECG confirming cardiac irregularity |

---

## Limitations

This project does not constitute a clinical trial and does not produce a validated medical device. The personal Apple Watch analysis is a single-subject case study and its findings cannot be generalised to the broader population. Consumer wearable data is derived from optical heart rate sensors, which measure cardiac activity through a fundamentally different mechanism than the clinical ECG recordings used for model training. All personal data findings are presented as exploratory and hypothesis-generating.

---

## Acknowledgements

Dataset provided by Physionet and the Computing in Cardiology Challenge 2017 organisers. Personal health data collected via Apple Watch SE (1st Generation) and exported through Apple Health.
