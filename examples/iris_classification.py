import logging, warnings, optuna
import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tmatlas.booleanisation.thermometer import Thermometer
from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier

import json
from tmatlas.inspectors.features import FeatureInspector
from tmatlas.inspectors.tmu_inspector import TMUInspector
from tmatlas.exporters.json import JsonExporter
from tmatlas.exporters.csv import export_data_to_csv, format_clause_activations

# --- housekeeping -----------------------------------------------------------
logging.basicConfig(level=logging.INFO)
optuna.logging.enable_default_handler()
optuna.logging.set_verbosity(optuna.logging.INFO)
warnings.filterwarnings("ignore")

# --- 1. load data -----------------------------------------------------------
iris = load_iris(
    as_frame=True
)  # feature names already included :contentReference[oaicite:0]{index=0}
X = iris.data  # 150 × 4 dataframe
y = iris.target.values  # numeric labels 0-2

feature_names = X.columns.get_level_values(0).tolist()

# --- 2. train / test split (stratified) -------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# --- 3. booleanise ----------------------------------------------------------
binarizer = Thermometer(max_bits_per_feature=10)
binarizer.fit(X_tr.to_numpy())
X_tr_tm = binarizer.transform(X_tr.to_numpy())
X_te_tm = binarizer.transform(X_te.to_numpy())


# --- 4. hyper-parameter search ---------------------------------------------
def objective(trial):
    params = {
        "number_of_clauses": trial.suggest_int("number_of_clauses", 20, 500),
        "T": trial.suggest_int("T", 10, 1000),
        "s": trial.suggest_float("s", 1.0, 10.0),
        "seed": trial.suggest_int("seed", 1, 10_000),
        "weighted_clauses": True,
    }
    tm = TMCoalescedClassifier(**params)
    tm.fit(X_tr_tm, y_tr, epochs=1)
    y_pred = tm.predict(X_te_tm)
    return 1.0 - accuracy_score(y_te, y_pred)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# --- 5. final training with best params -------------------------------------
best = study.best_trial.params
tm_best = TMCoalescedClassifier(**best, weighted_clauses=True)
tm_best.fit(X_tr_tm, y_tr, epochs=25)

# --- 6. evaluation ----------------------------------------------------------
y_pred_final = tm_best.predict(X_te_tm)
acc = accuracy_score(y_te, y_pred_final)
print(f"Best Optuna params: {best}")
print(f"Test accuracy: {acc:.4f}")

# --- 7. model saving ----------------------------------------------------------
y_pred_test_tm = tm_best.predict(X_te_tm)
y_pred_train_tm = tm_best.predict(X_tr_tm)


# ------------------ Model Saving with TMAtlas ------------------------------------

# Step 1: Build the feature inspector
feat_ext = FeatureInspector(binarizer, X_tr.to_numpy(), feature_names)

# Step 2: Create the inspector and For classification — pass class names
inspector = TMUInspector(tm_best, feat_ext, classes=iris.target_names.tolist())

# Step 3: Export
data = JsonExporter(inspector, feat_ext).export()

# Step 4: Save to file
output_filename = "tmu_coalesced_model.json"
with open(output_filename, "w") as f:
    json.dump(data, f, indent=2)

print(f"Model successfully exported to '{output_filename}'")

# 5. Get model outputs
clause_outputs = tm_best.transform(X_te_tm)

# 6. Find activated clauses
activated_clauses_list = format_clause_activations(clause_outputs)

# 7. Save input and model data to csv
export_data_to_csv(
    output_filename="final_model_output.csv",
    X_data=X_te,
    y_actual=y_te,
    y_predicted=y_pred_test_tm,
    activated_clauses_list=activated_clauses_list,
)
