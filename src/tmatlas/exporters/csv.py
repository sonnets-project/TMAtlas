"""CSV data exporter."""

import csv
from typing import Dict, List
from numpy.typing import NDArray

import numpy as np
import pandas as pd


def format_clause_activations(
    clause_activations: NDArray, key_name: str = "Activated_Clauses"
) -> List[Dict[str, List[int]]]:
    """
    Convert a 2D boolean activation matrix of clauses to a list of dicts.

    Args:
        clause_activations: Shape (n_samples, n_clauses), values 0/1.
        key_name: Dictionary key for each sample's activated clause list.
    Returns:
        One dict per sample mapping *key_name* -> list of active clause ids
    """
    if not isinstance(clause_activations, np.ndarray) or clause_activations.ndim != 2:
        raise ValueError("Clause activiations must be a 2D numpy array.")

    return [{key_name: np.where(row == 1)[0].tolist()} for row in clause_activations]


def export_data_to_csv(
    output_filename: str,
    X_data: pd.DataFrame,
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    activated_clauses_list: List[Dict[str, List[int]]],
) -> None:
    """
    Writes inputs, actualsm predictions, and clause activations to a CSV.

    Args:
        output_filename: Destination file path.
        X_data: Feature DataFrame (column names become CSV headers).
        y_actual: Ground-truth target values.
        y_predicted: Model predictions.
        activated_clauses_list: Per-sample activated clause dicts (see
            :func:`format_clause_activations`).
    """
    if not isinstance(X_data, pd.DataFrame):
        raise TypeError("X_data must be a pandas Dataframe.")

    feature_names = X_data.columns.tolist()

    clause_headers: set[str] = set()
    for item in activated_clauses_list:
        clause_headers.update(item.keys())
    sorted_clause_headers = sorted(clause_headers)

    headers = (
        ["Sample"] + feature_names + ["Actual", "Predicted"] + sorted_clause_headers
    )

    y_actual = y_actual.flatten()
    y_predicted = y_predicted.flatten()

    with open(output_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for i in range(len(X_data)):
            row: Dict[str, object] = {"Sample": i}
            row.update(X_data.iloc[i].to_dict())
            row["Actual"] = y_actual[i]
            row["Predicted"] = y_actual[i]

            if i < len(activated_clauses_list):
                for key, value in activated_clauses_list[i].items():
                    row[key] = ";".join(map(str, value))

            writer.writerow(row)

    print(f"\nData successfully exporter to '{output_filename}'")
