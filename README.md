# **LEVERAGING INTERPRETABLE MACHINE LEARNING FOR DROUGHT MONITORING IN THE CANARY ISLANDS: A DATA‑DRIVEN STUDY**

A modular Python framework for computing SPEI‑based drought indicators, engineering predictive features, training machine learning models, and generating interpretable drought forecasts across the Canary Islands. The workflow integrates Random Forest and XGBoost models with SHAP and LIME interpretability tools, producing transparent, data‑driven predictions of short‑term drought conditions.

## Input data
input datasets can be freely downloaded [here] (https://www.miteco.gob.es/ca/agua/temas/evaluacion-de-los-recursos-hidricos/evaluacion-recursos-hidricos-regimen-natural.html).

## How to run
> [!IMPORTANT]  
> This code is shared for transparency and does not run out‑of‑the‑box, as the training CSV files are not included in this repository.

First, make sure you have conda installed. Then install the environment and dependencies.

## 1. Clone the project:

```bash
git clone https://github.com/asadisara/SPEI_drought_index.git
```

## 2. Create and activate the environment

```bash
cd SPEI_drought_index
conda create -n spei-env python=3.11
conda activate spei-env
pip install -r requirements.txt
```

## 3. Add your data

```bash
data/
    your_SPEI_file.csv
```

This repository does not include datasets, so the code will not run until you add your own CSV files.

## 4. Run the full modelling pipeline

```bash
python main.py
```

This executes:

- feature engineering <br />

- model training and hyperparameter search <br />

- evaluation (train/test/full) <br />

- extrema and midpoint detection <br />

- SHAP + LIME interpretability <br />

- figure generation <br />

All outputs will be saved automatically to:

```bash
output/plots/
output/tables/
output/explanations/
```

## 5. (Optional) Clean outputs
If you want to clear previous results:

```bash
rm -r output/*
```
