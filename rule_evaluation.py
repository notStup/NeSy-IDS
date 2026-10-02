import sys
from io import StringIO

import numpy as np

from Performance_measures import (
    confusion_metrics_basic,
    Micro_calculate_measures,
    Macro_calculate_measures_basic,
)


# ============================================================
# CONFIGURAZIONE
# ============================================================

N_CLASSES = 2

# 0 = BENIGN
# 1 = ATTACK


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

    Le metriche finali sono calcolate sull'intero test set.
    """

    # ========================================================
    # CONVERSIONE INPUT
    # ========================================================

    if hasattr(x_test, "numpy"):
        x_test = x_test.numpy()

    x_test = np.asarray(x_test)

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

    if hasattr(ffcn_predictions, "numpy"):

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

    # validated_rules è già una lista piatta.
    rules = list(rules)

    print(
        "\nRegole utilizzate:",
        len(rules)
    )

    # ========================================================
    # ARRAY RISULTATI
    # ========================================================

    # Partiamo dalla predizione della FFCN.
    y_pred = ffcn_predictions.copy()

    # True quando almeno una ATTACK rule
    # è stata attivata sul campione.
    rule_covered_mask = np.zeros(
        len(y_test),
        dtype=bool
    )

    # Numero di override della FFCN
    # effettuati dalle regole.
    rule_override_count = 0

    # Numero di override sui campioni che
    # la FFCN aveva classificato BENIGN.
    benign_to_attack_count = 0

    # ========================================================
    # EVALUATION CAMPIONI
    # ========================================================

    for sample_index, x in enumerate(x_test):

        attack_rule_matches = False

        for original_rule in rules:

            # ------------------------------------------------
            # Le regole attive devono essere ATTACK rules.
            # ------------------------------------------------

            rule_label = int(
                original_rule[-1]
            )

            if rule_label != 1:
                continue

            # Copia della regola.
            rl = list(
                original_rule[:-1]
            )

            # ------------------------------------------------
            # Sostituzione feature index -> valore del flow
            # ------------------------------------------------

            i = 3

            while i < len(rl) - 1:

                try:

                    feature_index = int(
                        rl[i]
                    )

                    rl[i] = str(
                        x[feature_index]
                    )

                    i += 6

                except (
                    ValueError,
                    TypeError,
                    IndexError
                ):

                    break

            # ------------------------------------------------
            # Valutazione della regola
            # ------------------------------------------------

            flg_rule = False

            try:

                str_rule = (
                    "if ("
                    + " ".join(rl)
                    + "):\n"
                    + "\tprint(True)\n"
                    + "else:\n"
                    + "\tprint(False)"
                )

                old_stdout = sys.stdout
                result = StringIO()

                try:

                    sys.stdout = result
                    exec(str_rule)

                finally:

                    sys.stdout = old_stdout

                flg_rule = (
                    result.getvalue()
                    .replace("\n", "")
                ) == "True"

            except Exception as exc:

                print(
                    f"Errore valutando una regola: {exc}"
                )

                flg_rule = False

            # ------------------------------------------------
            # ATTACK RULE ATTIVATA
            # ------------------------------------------------

            if flg_rule:

                attack_rule_matches = True

                rule_covered_mask[
                    sample_index
                ] = True

                # Una sola ATTACK rule è sufficiente.
                break

        # ====================================================
        # DECISIONE IBRIDA
        # ====================================================

        if attack_rule_matches:

            # La componente simbolica rileva ATTACK.
            if y_pred[sample_index] == 0:
                benign_to_attack_count += 1

            if y_pred[sample_index] != 1:
                rule_override_count += 1

            y_pred[sample_index] = 1

        else:

            # Nessuna evidenza simbolica di ATTACK:
            # manteniamo la predizione della FFCN.
            y_pred[sample_index] = (
                ffcn_predictions[sample_index]
            )

    # ========================================================
    # COVERAGE COMPONENTE SIMBOLICA
    # ========================================================

    covered_count = int(
        np.sum(rule_covered_mask)
    )

    uncovered_count = int(
        np.sum(~rule_covered_mask)
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
    # METRICHE SISTEMA IBRIDO
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

    (
        precision,
        recall,
        f1
    ) = Macro_calculate_measures_basic(
        y_test,
        y_pred
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