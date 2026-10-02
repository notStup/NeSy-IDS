import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_dataset
from rule_diagnostics import (
    diagnose_rules,
    print_diagnostics,
    save_diagnostics,
)


# ============================================================
# CONFIGURAZIONE
# ============================================================

TRAIN_FRACTION = 0.001
VALIDATION_FRACTION = 0.10
TEST_SAMPLES = 5000
RANDOM_STATE = 42


RULES_PATH = (
    "Experiments/rule_set_initial.joblib"
)


# ============================================================
# CARICAMENTO DATASET
# ============================================================

X_train, X_test, y_train, y_test, _ = (
    preprocess_dataset()
)


# ============================================================
# RICOSTRUZIONE DELLO STESSO SPLIT
# USATO NEL MINI-ESPERIMENTO
# ============================================================

X_train, _, y_train, _ = train_test_split(
    X_train,
    y_train,
    train_size=TRAIN_FRACTION,
    random_state=RANDOM_STATE,
    stratify=y_train
)

X_train, X_val, y_train, y_val = (
    train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_FRACTION,
        random_state=RANDOM_STATE,
        stratify=y_train
    )
)

X_test, _, y_test, _ = train_test_split(
    X_test,
    y_test,
    train_size=TEST_SAMPLES,
    random_state=RANDOM_STATE,
    stratify=y_test
)


# ============================================================
# CONVERSIONE NUMERICA
# ============================================================

X_val = X_val.to_numpy()
y_val = y_val.to_numpy().reshape(-1)

X_test = X_test.to_numpy()
y_test = y_test.to_numpy().reshape(-1)


# ============================================================
# CARICAMENTO REGOLE VALIDATE
# ============================================================

rules = joblib.load(
    RULES_PATH
)

print(
    "Regole estratte prima della validation:",
    len(rules)
)


# ============================================================
# DIAGNOSTICA VALIDATION
# ============================================================

validation_report = diagnose_rules(
    rules,
    X_val,
    y_val,
    dataset_name="VALIDATION"
)

print_diagnostics(
    validation_report,
    "VALIDATION"
)

save_diagnostics(
    validation_report,
    "Experiments/rule_diagnostics_validation.csv"
)


# ============================================================
# DIAGNOSTICA TEST
# ============================================================

test_report = diagnose_rules(
    rules,
    X_test,
    y_test,
    dataset_name="TEST"
)

print_diagnostics(
    test_report,
    "TEST"
)

save_diagnostics(
    test_report,
    "Experiments/rule_diagnostics_test.csv"
)


# ============================================================
# CONFRONTO VALIDATION / TEST
# ============================================================

validation_report = (
    validation_report.set_index("rule_id")
)

test_report = (
    test_report.set_index("rule_id")
)

comparison = validation_report[
    [
        "class",
        "coverage",
        "coverage_rate",
        "benign_covered",
        "attack_covered",
        "correct",
        "errors",
        "precision",
    ]
].join(
    test_report[
        [
            "coverage",
            "coverage_rate",
            "benign_covered",
            "attack_covered",
            "correct",
            "errors",
            "precision",
        ]
    ],
    lsuffix="_validation",
    rsuffix="_test"
)

comparison.to_csv(
    "Experiments/rule_diagnostics_comparison.csv"
)

print(
    "\nReport confronto salvato in:",
    "Experiments/rule_diagnostics_comparison.csv"
)
