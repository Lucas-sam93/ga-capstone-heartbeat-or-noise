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
- 8,249 clinically validated single-lead ECG recordings (after exclusions)
- Binary label mapping: Normal (N) versus Abnormal (AF + Other rhythm)
- Source: https://physionet.org/content/challenge-2017/1.0.0/

### Secondary — Personal Apple Watch Data
- 5,485 heart rate variability readings spanning April 2021 to February 2026
- Supplementary metrics: resting heart rate, walking heart rate, continuous heart rate
- Clinical anchor point: ECG conducted June 2025 confirming cardiac irregularity
- Role: Consumer-grade signal bridge layer — not used for model training or validation

---

## Methodology

### Layer 1 — Clinical Benchmark
A binary cardiac rhythm classifier is trained and validated on the Physionet dataset using only features extractable from consumer wearable devices. This deliberate constraint ensures the model reflects real-world deployment conditions. Sensitivity to abnormal rhythms is the primary evaluation metric, consistent with the screening context in which missing a true positive carries greater consequence than generating a false alarm.

**Minimum performance threshold:** Sensitivity ≥ 0.80 at Specificity ≥ 0.75

### Layer 2 — Consumer Wearable Bridge
The trained model is applied to personal Apple Watch data to evaluate signal transferability from clinical ECG to consumer PPG-derived measurements. Success is assessed across three tiers:

- **Tier 1 — Foundational Applicability:** The feature pipeline executes on Apple Watch data with interpretable outputs and documented signal quality differences between clinical and consumer measurements.
- **Tier 2 — Signal Detection:** Model output scores show statistically meaningful deviation from personal baseline during the ninety-day window surrounding the June 2025 clinical ECG event.
- **Tier 3 — Clinical Concordance:** The ECG report confirms an irregularity consistent with the model's detection scope and the Apple Watch data demonstrates the Tier 2 deviation pattern.

---

## Project Structure
```
ga-capstone-heartbeat-or-noise/
│   README.md
│   requirements.txt
│   CLAUDE.md
│
├───data/
│   ├───physionet/        # 8,249 ECG recordings + REFERENCE-v3.csv
│   ├───apple_watch/      # Personal Apple Watch CSV exports
│   └───processed/        # Feature-engineered outputs
│
├───notebooks/
│   ├───01_data_exploration.ipynb
│   ├───02_feature_engineering.ipynb
│   ├───03_modelling.ipynb
│   ├───04_apple_watch_bridge.ipynb
│   └───05_conclusions.ipynb
│
├───src/
│   ├───features.py       # Feature extraction functions
│   ├───preprocess.py     # Signal cleaning functions
│   └───evaluate.py       # Model evaluation functions
│
└───outputs/
    ├───figures/
    └───models/
```

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
| Noisy class | Excluded | Signal quality failure, not rhythm classification |
| Classification type | Binary — Normal vs Abnormal | Screening requires triage decision, not diagnosis |
| Feature constraint | Wearable-extractable only | Ensures valid generalisation test |
| Primary metric | Sensitivity | Missing a true positive is more dangerous than a false positive in screening |
| Apple Watch role | Bridge layer only | N=1, no clinical outcome labels |

---

## Limitations

This project does not constitute a clinical trial and does not produce a validated medical device. The personal Apple Watch analysis is a single-subject case study and its findings cannot be generalised to the broader population. Consumer wearable data is derived from optical heart rate sensors, which measure cardiac activity through a fundamentally different mechanism than the clinical ECG recordings used for model training. All personal data findings are presented as exploratory and hypothesis-generating.

---

## Acknowledgements

Dataset provided by Physionet and the Computing in Cardiology Challenge 2017 organisers. Personal health data collected via Apple Watch and exported through Apple Health.
