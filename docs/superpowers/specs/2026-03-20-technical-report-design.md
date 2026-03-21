# Technical Report & Dashboard — Design Spec

**Date:** 2026-03-20
**Status:** Under Review
**Audience:** GA instructors + potential employers (mixed technical/non-technical)

---

## Overview

Three deliverables for the GA Capstone Project final submission:

1. **Jupyter notebook** (`notebooks/06_technical_report.ipynb`) — full narrative + code + figures
2. **DOCX report** (`docs/technical_report.docx`) — same narrative + figures, no code
3. **Streamlit dashboard** (`dashboard/app.py`) — interactive analytics + screening demo

All analysis is complete. These deliverables document and present existing results.

**BeatCheck** (`app/`) is a separate, pre-existing deliverable — the FastAPI-based screening web app. The Streamlit dashboard is an analytics companion, not a replacement. Both are project outputs.

**Execution dependency chain:** Notebook runs first (generates all figures) → DOCX script embeds figures → Dashboard loads pre-computed results and figures.

---

## Deliverable 1: Technical Report Notebook

**File:** `notebooks/06_technical_report.ipynb`

### Structure (GA Rubric Sections)

#### Section 1: Problem, Goals & Audience
- Singapore CVD public health context (30.5% of deaths, 49% unscreened)
- Research question (verbatim from CLAUDE.md)
- Success criteria: sensitivity >= 80%, specificity >= 75% (pre-registered)
- Target audience: general population screening via consumer wearables
- Project scope: what this IS and IS NOT

#### Section 2: Data Sources & Data Dictionary
- **Physionet 2017** — 8,528 ECG recordings, 300 Hz, 4 classes
  - Data dictionary table: column name, type, description, range, cleaning applied
  - After cleaning: 8,187 records (noisy excluded, quality filtered)
  - Binary mapping: N=0, A+O=1
- **Apple Watch** — 4.8 years personal data, 5 metrics
  - Data dictionary for each metric
  - Anchor period definitions
  - Device limitations (PPG only, no ECG)
- **MIMIC PERform AF** — 35 subjects, finger PPG, 125 Hz
  - Data dictionary: Time, PPG, ECG, resp columns
  - 19 AF + 16 NSR, pre-labelled

#### Section 3: Patterns, Trends & Insights
- Class distribution analysis (Physionet)
- Feature distributions Normal vs Abnormal (violin/box plots)
- Feature correlations (heatmap)
- Feature importance (Random Forest)
- Gap quantification — KS distances Physionet vs MIMIC
- Key insight: modality gap is central obstacle

#### Section 4: Predictive Model
- Feature engineering pipeline (8 locked features)
- Train/val/test split (80/10/10 stratified)
- Four models trained: LR, RF, XGBoost, SVM
- Cross-validation results (5-fold stratified)
- Model selection framework (five-criterion natural selection)
- SVM selected — highest specificity, no overfitting
- Layer 1 test results: sens 84.4%, spec 87.3%, AUROC 0.9080

##### 4a: Apple Watch N=1 Case Study
- Feature matrix: 1,997 windows across 4 anchor periods
- Gap quantification: 5 large, 3 moderate KS distances
- Mann-Whitney U tests: null result (no period significantly elevated)
- Tier assessment: Tier 1 PASS, Tier 2 FAIL, Tier 3 FAIL
- Clinical ECG report (June 2025, NHCS): intraventricular conduction delay — a waveform morphology abnormality structurally undetectable by HRV features
- Three mechanistic explanations for null result: modality gap, behavioural confound, abnormality type mismatch

##### 4b: MIMIC PERform AF Primary Validation
- 35 subjects, all green tier, AUROC 0.8586
- 100% sensitivity, 12.5% specificity at fixed threshold 0.34
- Cross-model comparison: all 4 models show same specificity failure pattern
- Gap quantification: all 8 features show large KS distances
- Threshold recalibration: Youden's J recovers specificity to 93.8%
- Stress testing: bootstrap AUROC 95% CI [0.7246, 0.9720]
- Sensitivity-targeted LOOCV: 78.9% sens / 81.2% spec (primary reported figure)
- LOOCV comparison: sensitivity-targeted vs Youden's J

##### 4c: BeatCheck Screening App
- FastAPI backend + clinical frontend
- Pipeline: 30-min windows, LOOCV mean threshold (0.8368), 3-tier risk classification
- Stress-tested with real Apple Health export

#### Section 5: Recommendations & Next Steps
- Answer to research question (qualified yes — discriminative signal transfers, threshold recalibration needed)
- Clinical implications for Singapore screening
- Limitations (N=35 validation, single PPG modality, ICU population bias)
- Recommendations:
  - Domain-adapted threshold for wearable deployment
  - Larger multi-site PPG validation study
  - Multi-lead or hybrid feature approaches
  - Regulatory pathway considerations
- Setbacks documented (Apple Watch null result, Simband access, N=35 constraint)

### Figure Generation
All figures generated inline in the notebook, saved to `outputs/figures/`, logged in `figures_log.csv`. Existing figures (`sample_ecg.png`, `aw_vs_physionet_distributions.png`) retained — new figures complement, not replace.

---

## Deliverable 2: DOCX Report

**File:** `docs/technical_report.docx`
**Generator:** `scripts/generate_report_docx.py`

### Format
- Title page: project title, author (Lucas Sam), date, GA branding
- Table of contents
- Numbered sections matching notebook structure
- Figures embedded with captions and figure numbers
- Tables formatted professionally
- No code blocks
- Footer with page numbers

### Generation approach
- `python-docx` library
- Narrative text written as structured content in the script (mirrors notebook markdown sections)
- Figures loaded from `outputs/figures/`
- Single command: `python scripts/generate_report_docx.py`
- Note: narrative changes require updating both notebook and script. Accepted trade-off for clean DOCX formatting control.

---

## Deliverable 3: Streamlit Dashboard

**File:** `dashboard/app.py` (separate directory from BeatCheck's `app/`)

### Tabs

#### Tab 1: Model Performance
- Layer 1 metrics summary (sensitivity, specificity, AUROC, F1)
- Interactive ROC curve (Plotly) — pre-computed from Layer 1 test set predictions
- Confusion matrix visualisation
- Feature importance bar chart
- Threshold slider operating on pre-computed ROC curve points (Layer 1 test data) — visual only, no retraining

#### Tab 2: Cross-Model Comparison
- Side-by-side metrics table (all 4 models)
- Grouped bar chart comparing models
- Overfitting analysis (train vs test gap)
- Selection rationale display

#### Tab 3: Layer 2 Validation
- MIMIC PERform AF results summary
- Gap quantification bar chart (KS distances)
- Probability score distribution (AF vs NSR)
- Threshold recalibration before/after
- Cross-model comparison on MIMIC (all 4 models at fixed thresholds)
- Stress test results (bootstrap AUROC histogram)
- LOOCV comparison (Youden's J vs sensitivity-targeted)

#### Tab 4: Screening Demo
- File upload (CSV or XML)
- Imports inference functions from `app/pipeline.py` (confirmed: no FastAPI-specific dependencies in the public functions — they accept bytes/DataFrames and return dicts)
- Displays: risk tier, percentage flagged, windows analysed, days covered
- Disclaimer banner (same as BeatCheck)

### Technical Details
- Loads saved model artefacts from `app/models/`
- Loads pre-computed results from `outputs/` CSVs (no retraining)
- Figures rendered with Plotly for interactivity
- `streamlit` and `plotly` added to project requirements

---

## Figure Specifications

### Design Principles
- Titles as questions or plain statements
- Colour palette: green (#2ecc71) = Normal, red/orange (#e74c3c / #f39c12) = Abnormal, blue (#3498db) = neutral
- Large fonts (title 16pt, labels 12pt, ticks 10pt)
- Minimal gridlines, clean whitespace
- Annotations for key takeaways directly on figure
- 150 dpi minimum, saved as PNG
- No abbreviations without explanation

### Figure List (15 figures)

| # | Filename | Title | Type | Notes |
|---|---|---|---|---|
| 1 | class_distribution_2026.png | What Does Our Training Data Look Like? | Stacked bar | |
| 2 | pipeline_flowchart_2026.png | How the Screening System Works | Flowchart | matplotlib patches + arrows (no external diagramming lib) |
| 3 | feature_descriptions_2026.png | What Health Signals Does the Model Use? | Horizontal bar | |
| 4 | feature_distributions_by_class_2026.png | How Do Healthy and Unhealthy Hearts Differ? | Violin/box | |
| 5 | feature_correlation_heatmap_2026.png | Which Measurements Move Together? | Heatmap | |
| 6 | model_comparison_scorecard_2026.png | Which Model Performed Best? | Grouped bar | |
| 7 | roc_curve_svm_layer1_2026.png | How Well Does the Model Separate Healthy from Unhealthy? | ROC curve | |
| 8 | confusion_matrix_svm_layer1_2026.png | Where Does the Model Get It Right and Wrong? | Quadrant | Plain labels: Correctly Flagged, Missed, False Alarm, Correctly Cleared |
| 9 | feature_importance_2026.png | Which Measurements Matter Most? | Horizontal bar | |
| 10 | modality_gap_ks_distances_2026.png | How Different Is Wearable Data from Clinical Data? | Grouped bar | |
| 11 | mimic_probability_scores_2026.png | Can the Model Tell AF from Normal Using Wearable Data? | Strip/dot plot | |
| 12 | threshold_recalibration_2026.png | What Happens When We Adjust for Wearable Data? | Before/after bars | Two side-by-side grouped bars, not confusion matrices |
| 13 | bootstrap_auroc_distribution_2026.png | How Confident Are We in the Model? | Histogram | |
| 14 | loocv_comparison_2026.png | Which Threshold Strategy Works Better for Screening? | Side-by-side bar | |
| 15 | risk_tier_explanation_2026.png | What Do the Screening Results Mean? | Infographic | matplotlib text + coloured rectangles (no external tools) |

### Existing Figures (Retained)
- `sample_ecg.png` — referenced in Section 2 (Physionet data)
- `aw_vs_physionet_distributions.png` — referenced in Section 4a (Apple Watch gap)

---

## Dependencies

### New packages needed
- `python-docx` — DOCX generation
- `streamlit` — dashboard
- `plotly` — interactive charts in Streamlit

### Existing packages used
- `matplotlib`, `seaborn` — static figures
- `pandas`, `numpy` — data handling
- `scikit-learn` — model loading and metrics
- `joblib` — model artefact loading

---

## File Structure (New Files)

```
GA Capstone Project/
├── notebooks/
│   └── 06_technical_report.ipynb          # NEW — master report
├── docs/
│   └── technical_report.docx              # NEW — generated DOCX
├── scripts/
│   └── generate_report_docx.py            # NEW — DOCX generator
├── dashboard/
│   └── app.py                             # NEW — Streamlit dashboard
└── outputs/
    └── figures/                           # 15 new PNGs + 2 existing
```

### Relationship to Existing Files
- `app/` — BeatCheck (FastAPI). Untouched. Separate deliverable.
- `dashboard/app.py` imports from `app/pipeline.py` for the screening demo tab.
- `notebooks/01-05` — analysis notebooks. Untouched. Referenced in report narrative.

---

## GA Rubric Coverage

| Rubric Criterion | Report Section | Deliverable |
|---|---|---|
| Problem, Goals and Audiences | Section 1 | Notebook + DOCX |
| Data Sources and Definitions in a Data Dictionary | Section 2 | Notebook + DOCX |
| Patterns, Trends and Insights | Section 3 | Notebook + DOCX + Dashboard (Tabs 1-3) |
| At least one interactive dashboard visualisation | — | Dashboard (all tabs) + BeatCheck app |
| At least one predictive model | Section 4 | Notebook + DOCX + Dashboard (Tab 4) |
| Recommendations | Section 5 | Notebook + DOCX |
