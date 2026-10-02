import numpy as np
import pandas as pd


# ============================================================
# PARSING CAPSRule
# ============================================================

def _to_float(value):
    """Converte i valori serializzati da CapsRule in float."""
    if isinstance(value, str):
        value = value.strip()

        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1].strip()

        return float(value)

    return float(
        np.asarray(value).reshape(-1)[0]
    )


def _to_feature_index(value):
    """Converte l'indice della feature in int."""
    return int(
        np.asarray(value).reshape(-1)[0]
    )


def _parse_condition_block(tokens):
    """
    Converte un blocco CapsRule del tipo:

        (
            lower <= feature <= upper
            and
            lower <= feature <= upper
        )

    in:

        [(feature, lower, upper), ...]
    """

    tokens = list(tokens)

    if tokens and tokens[0] == "(":
        tokens = tokens[1:]

    if tokens and tokens[-1] == ")":
        tokens = tokens[:-1]

    conditions = []
    i = 0

    while i < len(tokens):

        if i + 4 >= len(tokens):
            raise ValueError(
                f"Blocco regola incompleto: {tokens}"
            )

        lower = _to_float(tokens[i])

        if tokens[i + 1] != "<=":
            raise ValueError(
                "Operatore <= non trovato nel blocco regola."
            )

        feature = _to_feature_index(
            tokens[i + 2]
        )

        if tokens[i + 3] != "<=":
            raise ValueError(
                "Secondo operatore <= non trovato nel blocco regola."
            )

        upper = _to_float(
            tokens[i + 4]
        )

        conditions.append(
            (feature, lower, upper)
        )

        i += 5

        if i < len(tokens):

            if tokens[i] == "and":
                i += 1

            else:
                raise ValueError(
                    f"Token inatteso nel blocco regola: {tokens[i]}"
                )

    return conditions


def parse_rule(rule):
    """
    Converte una singola regola aggregata CapsRule in una
    struttura leggibile.

    Una regola aggregata può contenere più rami OR.

    Esempio:

        (
            A and B
        ) or (
            C
        )

    diventa:

        {
            "class": 1,
            "branches": [
                [...],
                [...]
            ]
        }
    """

    rule = list(rule)

    if not rule:
        raise ValueError("Regola vuota.")

    rule_class = int(rule[-1])
    tokens = rule[:-1]

    branches = []
    current = []

    for token in tokens:

        if token == ") or (":

            if current:
                branches.append(
                    _parse_condition_block(current)
                )

            current = []

        else:

            current.append(token)

    if current:
        branches.append(
            _parse_condition_block(current)
        )

    return {
        "class": rule_class,
        "branches": branches
    }


def flatten_rules(rules):
    """
    Accetta sia:

        [rule1, rule2, ...]

    sia:

        [
            [rule1, rule2],
            [rule3, rule4],
            ...
        ]

    e restituisce una lista piatta.
    """

    rules = list(rules)

    if not rules:
        return []

    first = rules[0]

    # Una regola CapsRule termina con una label numerica.
    if (
        isinstance(first, (list, tuple, np.ndarray))
        and len(first) > 0
        and np.isscalar(first[-1])
        and not isinstance(first[-1], str)
    ):
        return rules

    flattened = []

    for group in rules:

        flattened.extend(group)

    return flattened


def parse_rules(rules):
    """Converte tutte le regole in formato strutturato."""

    return [
        parse_rule(rule)
        for rule in flatten_rules(rules)
    ]


# ============================================================
# MATCH DI UNA REGOLA
# ============================================================

def branch_matches(branch, x):
    """True se tutte le condizioni del ramo sono soddisfatte."""

    for feature, lower, upper in branch:

        value = float(x[feature])

        if value < lower or value > upper:
            return False

    return True


def rule_matches(rule, x):
    """
    Una regola aggregata è soddisfatta se almeno uno dei suoi
    rami OR è soddisfatto.
    """

    return any(
        branch_matches(branch, x)
        for branch in rule["branches"]
    )


# ============================================================
# DIAGNOSTICA PER REGOLA
# ============================================================

def diagnose_rules(
    rules,
    X,
    y,
    dataset_name="dataset"
):
    """
    Calcola, per ogni regola:

        rule_id
        class
        branches
        conditions
        coverage
        benign_covered
        attack_covered
        correct
        errors
        precision
        error_rate_covered

    La coverage indica quanti campioni sono coperti dalla
    regola.

    La precision misura quanti campioni coperti appartengono
    alla classe prevista dalla regola.

    L'error_rate_covered è calcolato soltanto sui campioni
    coperti dalla regola.
    """

    structured_rules = parse_rules(rules)

    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    rows = []

    for rule_id, rule in enumerate(structured_rules):

        covered_mask = np.zeros(
            len(X),
            dtype=bool
        )

        for index, x in enumerate(X):

            covered_mask[index] = (
                rule_matches(
                    rule,
                    x
                )
            )

        covered_y = y[
            covered_mask
        ]

        coverage = int(
            np.sum(covered_mask)
        )

        benign_covered = int(
            np.sum(covered_y == 0)
        )

        attack_covered = int(
            np.sum(covered_y == 1)
        )

        correct = int(
            np.sum(
                covered_y
                == rule["class"]
            )
        )

        errors = (
            coverage
            - correct
        )

        precision = (
            correct / coverage
            if coverage > 0
            else 0.0
        )

        error_rate_covered = (
            errors / coverage
            if coverage > 0
            else 0.0
        )

        conditions = sum(
            len(branch)
            for branch
            in rule["branches"]
        )

        rows.append(
            {
                "dataset": dataset_name,
                "rule_id": rule_id,
                "class": rule["class"],
                "branches": len(rule["branches"]),
                "conditions": conditions,
                "coverage": coverage,
                "coverage_rate":
                    coverage / len(X)
                    if len(X) > 0
                    else 0.0,
                "benign_covered":
                    benign_covered,
                "attack_covered":
                    attack_covered,
                "correct": correct,
                "errors": errors,
                "precision": precision,
                "error_rate_covered":
                    error_rate_covered
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SUMMARY
# ============================================================

def print_diagnostics(
    report,
    dataset_name
):
    """Stampa una diagnostica leggibile."""

    print(
        f"\n=== RULE DIAGNOSTICS: {dataset_name} ==="
    )

    if report.empty:

        print("Nessuna regola.")
        return

    for _, row in report.iterrows():

        print(
            f"R{int(row['rule_id']):03d} | "
            f"class={int(row['class'])} | "
            f"branches={int(row['branches'])} | "
            f"conditions={int(row['conditions'])} | "
            f"coverage={int(row['coverage'])} "
            f"({row['coverage_rate']:.3f}) | "
            f"BENIGN={int(row['benign_covered'])} | "
            f"ATTACK={int(row['attack_covered'])} | "
            f"correct={int(row['correct'])} | "
            f"errors={int(row['errors'])} | "
            f"precision={row['precision']:.3f}"
        )

    print(
        "\nRegole classe BENIGN:",
        int(
            np.sum(
                report["class"] == 0
            )
        )
    )

    print(
        "Regole classe ATTACK:",
        int(
            np.sum(
                report["class"] == 1
            )
        )
    )


# ============================================================
# SALVATAGGIO
# ============================================================

def save_diagnostics(
    report,
    path
):
    """Salva il report diagnostico in CSV."""

    report.to_csv(
        path,
        index=False
    )
