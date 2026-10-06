import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from Performance_measures import (
    confusion_metrics_basic,
    Micro_calculate_measures,
)


# ============================================================
# CONFIGURAZIONE
# ============================================================

N_CLASSES = 2

# 0 = BENIGN
# 1 = ATTACK
ATTACK_CLASS = 1


# ============================================================
# CONVERSIONE SCALARE
# ============================================================

def to_scalar(value):
    """
    Converte un valore eventualmente rappresentato come:

        12.5
        [12.5]
        np.array([12.5])
        np.array([[12.5]])

    in un float.
    """

    if isinstance(value, str):

        value = value.strip()

        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1].strip()

    array = np.asarray(
        value,
        dtype=float
    ).reshape(-1)

    if len(array) == 0:
        raise ValueError(
            "Impossibile convertire il valore in scalare."
        )

    return float(array[0])


# ============================================================
# MATCHING DELLA REGOLA
# ============================================================

def rule_matches(rule, sample):
    """
    Verifica se un campione soddisfa una regola.

    Formato atteso:

        [
            "(",
            min_value,
            "<=",
            feature_name,
            "Feature:",
            feature_index,
            "<=",
            max_value,
            "and",
            ...,
            ")",
            class_label
        ]

    Esempio:

        [
            "(",
            "1548.67",
            "<=",
            "Bwd Packet Length Std",
            "Feature:",
            20,
            "<=",
            "5790.16",
            ")",
            1
        ]

    corrisponde a:

        1548.67 <= sample[20] <= 5790.16
    """

    i = 1

    while i < len(rule) - 1:

        # Fine della regola
        if rule[i] == ")":
            break

        # ----------------------------------------------------
        # Struttura della condizione
        #
        # i     = minimum
        # i + 1 = <=
        # i + 2 = feature name
        # i + 3 = Feature:
        # i + 4 = feature index
        # i + 5 = <=
        # i + 6 = maximum
        # ----------------------------------------------------

        try:

            minimum = to_scalar(
                rule[i]
            )

            feature_index = int(
                rule[i + 4]
            )

            maximum = to_scalar(
                rule[i + 6]
            )

            value = to_scalar(
                sample[feature_index]
            )

        except (
            ValueError,
            TypeError,
            IndexError
        ):

            return False

        # ----------------------------------------------------
        # Valutazione intervallo
        # ----------------------------------------------------

        if not (
            minimum
            <= value
            <= maximum
        ):

            return False

        # ----------------------------------------------------
        # Passiamo alla condizione successiva
        # ----------------------------------------------------

        i += 7

        if (
            i < len(rule) - 1
            and rule[i] == "and"
        ):

            i += 1

    return True


# ============================================================
# EVALUATION DELLE REGOLE
# ============================================================

def evaluate_rules_boundary(
    rules,
    x_test,
    y_test_one,
    ffcn_predictions=None
):
    """
    Valutazione ibrida:

        ATTACK rule match -> ATTACK
        nessun ATTACK rule match -> predizione FFCN

    La coverage misura la percentuale di campioni
    sui quali almeno una ATTACK rule viene attivata.
    """

    # ========================================================
    # CONVERSIONE INPUT
    # ========================================================

    if hasattr(x_test, "numpy"):
        x_test = x_test.numpy()

    x_test = np.asarray(
        x_test
    )

    # ========================================================
    # CONVERSIONE LABEL
    # ========================================================

    y_test = np.asarray(
        y_test_one
    ).astype(int)

    if y_test.ndim == 2:

        if y_test.shape[1] == 1:

            y_test = y_test[:, 0]

        else:

            y_test = np.argmax(
                y_test,
                axis=1
            )

    if y_test.ndim != 1:

        raise ValueError(
            "y_test deve contenere etichette 1D "
            "oppure etichette one-hot 2D."
        )

    # ========================================================
    # PREDIZIONI FFCN
    # ========================================================

    if ffcn_predictions is None:

        raise ValueError(
            "ffcn_predictions è obbligatorio "
            "per la valutazione ibrida."
        )

    if hasattr(
        ffcn_predictions,
        "numpy"
    ):

        ffcn_predictions = (
            ffcn_predictions.numpy()
        )

    ffcn_predictions = np.asarray(
        ffcn_predictions
    ).astype(int).reshape(-1)

    if len(ffcn_predictions) != len(y_test):

        raise ValueError(
            "ffcn_predictions e y_test "
            "devono avere la stessa lunghezza."
        )

    # ========================================================
    # REGOLE
    # ========================================================

    rules = list(rules)

    print(
        "\nRegole utilizzate:",
        len(rules)
    )

    # ========================================================
    # ARRAY RISULTATI
    # ========================================================

    # Partiamo dalle predizioni FFCN.
    y_pred = ffcn_predictions.copy()

    # True se almeno una ATTACK rule
    # viene attivata.
    rule_covered_mask = np.zeros(
        len(y_test),
        dtype=bool
    )

    # Numero totale di override FFCN.
    rule_override_count = 0

    # Numero di casi:
    # FFCN = BENIGN
    # Rule = ATTACK
    benign_to_attack_count = 0

    # ========================================================
    # EVALUATION CAMPIONI
    # ========================================================

    for sample_index, sample in enumerate(
        x_test
    ):

        attack_rule_matches = False

        # ----------------------------------------------------
        # Proviamo tutte le ATTACK rules
        # ----------------------------------------------------

        for rule in rules:

            # ------------------------------------------------
            # Consideriamo solo ATTACK rules
            # ------------------------------------------------

            if (
                not isinstance(rule, (list, tuple))
                or len(rule) < 2
            ):

                continue

            try:

                rule_label = int(
                    rule[-1]
                )

            except (
                ValueError,
                TypeError
            ):

                continue

            if rule_label != ATTACK_CLASS:
                continue

            # ------------------------------------------------
            # Matching diretto
            # ------------------------------------------------

            if rule_matches(
                rule,
                sample
            ):

                attack_rule_matches = True

                rule_covered_mask[
                    sample_index
                ] = True

                # Una sola ATTACK rule
                # è sufficiente.
                break

        # ====================================================
        # DECISIONE IBRIDA
        # ====================================================

        if attack_rule_matches:

            # La componente simbolica
            # segnala ATTACK.

            if (
                ffcn_predictions[sample_index]
                == 0
            ):

                benign_to_attack_count += 1
                rule_override_count += 1

            y_pred[sample_index] = ATTACK_CLASS

        else:

            # Nessuna ATTACK rule:
            # manteniamo la FFCN.

            y_pred[sample_index] = (
                ffcn_predictions[sample_index]
            )

    # ========================================================
    # COVERAGE
    # ========================================================

    covered_count = int(
        np.sum(
            rule_covered_mask
        )
    )

    uncovered_count = int(
        np.sum(
            ~rule_covered_mask
        )
    )

    coverage = (
        covered_count / len(y_test)
        if len(y_test) > 0
        else 0.0
    )

    print(
        "\n=== COVERAGE ATTACK RULES ==="
    )

    print(
        "Test samples:",
        len(y_test)
    )

    print(
        "Rule-covered:",
        covered_count
    )

    print(
        "No rule match:",
        uncovered_count
    )

    print(
        "Coverage:",
        f"{coverage:.4f}"
    )

    print(
        "Rule overrides FFCN:",
        rule_override_count
    )

    print(
        "BENIGN -> ATTACK overrides:",
        benign_to_attack_count
    )

    # ========================================================
    # METRICHE HYBRID
    # ========================================================

    (
        mcm,
        tp_mean,
        tn_mean,
        fp_mean,
        fn_mean
    ) = confusion_metrics_basic(
        y_test,
        y_pred
    )

    out_measures = Micro_calculate_measures(
        tp_mean,
        tn_mean,
        fp_mean,
        fn_mean,
        0
    )

    # --------------------------------------------------------
    # ATTACK precision / recall / F1
    # --------------------------------------------------------

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=ATTACK_CLASS,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=ATTACK_CLASS,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=ATTACK_CLASS,
        zero_division=0
    )

    print(
        "\n=== HYBRID FFCN + ATTACK RULES RESULTS ==="
    )

    print(
        "Precision:",
        f"{precision:.4f}"
    )

    print(
        "Recall:",
        f"{recall:.4f}"
    )

    print(
        "F1:",
        f"{f1:.4f}"
    )

    # ========================================================
    # SALVATAGGIO RISULTATI
    # ========================================================

    np.savetxt(
        "Experiments/Micro_Test_Conf_Hybrid.csv",
        mcm,
        delimiter=",",
        fmt="%s"
    )

    np.savetxt(
        "Experiments/Micro_Test_Measures_Hybrid.csv",
        out_measures.to_numpy(),
        delimiter=",",
        fmt="%s"
    )

    with open(
        "Experiments/Macro_Test_Results_Hybrid.txt",
        "w"
    ) as file:

        file.write(
            "PR:" + str(precision)
            + " RC:" + str(recall)
            + " F1:" + str(f1)
        )

    np.savetxt(
        "Experiments/y_true_Hybrid_Test.csv",
        y_test,
        delimiter=","
    )

    np.savetxt(
        "Experiments/y_pre_Hybrid_Test.csv",
        y_pred,
        delimiter=","
    )

    return y_pred