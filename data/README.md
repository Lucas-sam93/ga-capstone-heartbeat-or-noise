# Data Directory

This project uses three data sources. Raw data files are excluded from the repository due to size and licensing constraints. Follow the instructions below to reproduce the full dataset locally.

---

## 1. PhysioNet 2017 AF Challenge (`data/physionet/`)

**License:** Open Data Commons Attribution License (ODC-BY)
**Access:** Public — no registration required

**Download:**

```bash
wget -r -N -c -np https://physionet.org/files/challenge-2017/1.0.0/ -P data/physionet/
```

Or via the PhysioNet web interface:
- Go to: https://physionet.org/content/challenge-2017/1.0.0/
- Download the full dataset (training set: 8,528 `.mat` + `.hea` files)
- Place all files in `data/physionet/`

**Expected files after download:**
- `A00001.mat` through `A08528.mat` — raw ECG recordings (MATLAB format)
- `A00001.hea` through `A08528.hea` — header files with metadata
- `RECORDS` — index of all recording IDs (already in repo)
- `REFERENCE.csv` — rhythm labels per recording (already in repo)

---

## 2. MIMIC PERform AF Dataset (`data/mimic_perform_af/`)

**License:** PhysioNet Credentialed Health Data License 1.5.0
**Access:** Requires free registration + data use agreement at PhysioNet

**Steps to access:**
1. Register at: https://physionet.org/register/
2. Complete the required CITI training
3. Request access to: https://physionet.org/content/mimic-perform-af/1.0.0/
4. Once approved, download via:

```bash
wget -r -N -c -np --user <your-physionet-username> --ask-password \
  https://physionet.org/files/mimic-perform-af/1.0.0/ \
  -P data/mimic_perform_af/
```

**Expected structure after download:**
```
data/mimic_perform_af/
  mimic_perform_af_csv/        # 19 AF subject CSV files
  mimic_perform_non_af_csv/    # 16 NSR subject CSV files
```

**Note:** This dataset may not be redistributed. Do not commit these files to any public repository.

---

## 3. Apple Watch Data (`data/apple_watch/`, `data/processed/`)

This is personal health data and is not reproducible by others. The Apple Watch analysis (Layer 2A) uses a single subject's export from the Apple Health app.

If you wish to reproduce this layer with your own data:
1. Export your Apple Health data: Settings → [Your Name] → Export All Health Data
2. Place `export.xml` in `data/apple_watch/`
3. Run the preprocessing notebooks in order

---

## What is already in the repository

The following files are committed and require no download:

| File | Description |
|---|---|
| `data/physionet/RECORDS` | Index of all PhysioNet recording IDs |
| `data/physionet/REFERENCE.csv` | Ground-truth rhythm labels |
| `data/physionet/REFERENCE-v3.csv` | Updated labels (v3 revision) |
| `data/processed/physionet_features.csv` | Extracted HRV features from PhysioNet (Layer 1 input) |
| `outputs/models/selected_model.joblib` | Final trained SVM model |
| `outputs/models/scaler.joblib` | Feature scaler (required for inference) |
| `outputs/models/evaluation_report.json` | Layer 1 evaluation metrics |
