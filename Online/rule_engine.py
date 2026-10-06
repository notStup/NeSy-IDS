import numpy as np


# ============================================================
# CONVERSIONE VALORI
# ============================================================

def _to_float(value):
    """
    Converte un valore proveniente dal formato di output
    di CapsRule in float.

    Esempi:
        '[20.]'            -> 20.0
        '[3.4641016]'      -> 3.4641016
        np.float32(20.0)   -> 20.0
    """

    if isinstance(value, str):

        value = value.strip()

        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1].strip()

        return float(value)

    return float(
        np.asarray(value).reshape(-1)[0]
    )


def _to_feature_index(value):
    """
    Converte l'indice della feature in int.
    """

    return int(
        np.asarray(value).reshape(-1)[0]
    )


# ============================================================
# PARSING DI UNA REGOLA CAPSRule
# ============================================================

def _parse_condition_block(tokens):
    """
    Converte un blocco del tipo:

        (
            lower <= feature <= upper
            and
            lower <= feature <= upper
        )

    in:

        [
            (feature, lower, upper),
            ...
        ]
    """

    tokens = list(tokens)

    # Rimuoviamo eventuali parentesi esterne.
    if tokens and tokens[0] == "(":
        tokens = tokens[1:]

    if tokens and tokens[-1] == ")":
        tokens = tokens[:-1]

    conditions = []

    i = 0

    while i < len(tokens):

        lower = _to_float(
            tokens[i]
        )

        i += 1

        if tokens[i] != "<=":
            raise ValueError(
                "Formato regola non valido: "
                "operatore <= atteso."
            )

        i += 1

        feature = _to_feature_index(
            tokens[i]
        )

        i += 1

        if tokens[i] != "<=":
            raise ValueError(
                "Formato regola non valido: "
                "secondo <= atteso."
            )

        i += 1

        upper = _to_float(
            tokens[i]
        )

        i += 1

        conditions.append(
            (
                feature,
                lower,
                upper
            )
        )

        # Eventuale AND.
        if i < len(tokens) and tokens[i] == "and":
            i += 1

    return conditions


# ============================================================
# CONVERSIONE DELLE REGOLE CAPSRule
# ============================================================

def parse_capsrule_groups(rule_groups):
    """
    Converte il formato prodotto da extract_rules_boundary():

        [
            ['(', ..., ') or (', ..., ')', 0],
            ['(', ..., ') or (', ..., ')', 1]
        ]

    in singole regole strutturate:

        [
            {
                'class': 0,
                'conditions': [...]
            },
            ...
        ]
    """

    structured_rules = []

    for grouped_rule in rule_groups:

        if not grouped_rule:
            continue

        rule_class = int(
            grouped_rule[-1]
        )

        tokens = list(
            grouped_rule[:-1]
        )

        current_block = []

        for token in tokens:

            # CapsRule usa ') or (' come singolo token.
            if token == ") or (":

                if current_block:
                    structured_rules.append(
                        {
                            "class": rule_class,
                            "conditions":
                                _parse_condition_block(
                                    current_block
                                )
                        }
                    )

                current_block = []

            else:

                current_block.append(token)

        # Ultimo blocco.
        if current_block:

            structured_rules.append(
                {
                    "class": rule_class,
                    "conditions":
                        _parse_condition_block(
                            current_block
                        )
                }
            )

    return structured_rules


def parse_rule_set(rule_set):
    """
    Converte il rule_set completo prodotto da più batch.

    Input:

        [
            [regola_classe_0_batch1,
             regola_classe_1_batch1],

            [regola_classe_0_batch2,
             regola_classe_1_batch2],
            ...
        ]

    Output:

        [
            singola_regola_1,
            singola_regola_2,
            ...
        ]
    """

    structured_rules = []

    for batch_rules in rule_set:

        structured_rules.extend(
            parse_capsrule_groups(
                batch_rules
            )
        )

    return structured_rules


# ============================================================
# MATCH DI UNA REGOLA
# ============================================================

def rule_matches(rule, x):
    """
    Verifica se un singolo flow soddisfa tutte le condizioni
    della regola.
    """

    for feature, lower, upper in rule["conditions"]:

        value = x[feature]

        if value < lower or value > upper:
            return False

    return True


# ============================================================
# CLASSIFICAZIONE CON LE REGOLE
# ============================================================

def classify_with_rules(rules, x):
    """
    Classifica un singolo flow usando le regole.

    Returns
    -------
    decision : int | None
        Classe predetta, oppure None se nessuna regola
        copre il flow.

    matched_rules : list[int]
        Indici delle regole che hanno coperto il flow.
    """

    votes = np.zeros(
        2,
        dtype=int
    )

    matched_rules = []

    for rule_index, rule in enumerate(rules):

        if rule_matches(
            rule,
            x
        ):

            matched_rules.append(
                rule_index
            )

            votes[
                rule["class"]
            ] += 1

    if not matched_rules:

        return None, []

    decision = int(
        np.argmax(votes)
    )

    return decision, matched_rules


# ============================================================
# CLASSIFICAZIONE DI UN DATASET
# ============================================================

def predict_with_rules(rules, X):
    """
    Applica le regole a tutti i flow.
    """

    X = np.asarray(X)

    predictions = np.full(
        len(X),
        -1,
        dtype=int
    )

    covered = np.zeros(
        len(X),
        dtype=bool
    )

    matched_rules = []

    for i, x in enumerate(X):

        decision, matches = (
            classify_with_rules(
                rules,
                x
            )
        )

        if decision is not None:

            predictions[i] = decision
            covered[i] = True

        matched_rules.append(
            matches
        )

    return (
        predictions,
        covered,
        matched_rules
    )


# ============================================================
# ANALISI DELLE SINGOLE REGOLE
# ============================================================

def analyze_rules(
    rules,
    X,
    y
):
    """
    Analizza ogni regola individualmente.

    Per ogni regola restituisce:

        coverage
        correct
        errors
        precision
        benign_covered
        attack_covered
    """

    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    report = []

    for rule_index, rule in enumerate(rules):

        mask = np.ones(
            len(X),
            dtype=bool
        )

        for feature, lower, upper in (
            rule["conditions"]
        ):

            mask &= (
                X[:, feature] >= lower
            )

            mask &= (
                X[:, feature] <= upper
            )

        covered_count = int(
            np.sum(mask)
        )

        if covered_count == 0:

            report.append(
                {
                    "rule_id": rule_index,
                    "class": rule["class"],
                    "conditions":
                        len(rule["conditions"]),
                    "coverage": 0,
                    "correct": 0,
                    "errors": 0,
                    "precision": 0.0,
                    "benign_covered": 0,
                    "attack_covered": 0
                }
            )

            continue

        covered_labels = y[mask]

        correct = int(
            np.sum(
                covered_labels
                == rule["class"]
            )
        )

        errors = (
            covered_count
            - correct
        )

        precision = (
            correct
            / covered_count
        )

        report.append(
            {
                "rule_id": rule_index,
                "class": rule["class"],
                "conditions":
                    len(rule["conditions"]),
                "coverage": covered_count,
                "correct": correct,
                "errors": errors,
                "precision": precision,
                "benign_covered": int(
                    np.sum(
                        covered_labels == 0
                    )
                ),
                "attack_covered": int(
                    np.sum(
                        covered_labels == 1
                    )
                )
            }
        )

    return report