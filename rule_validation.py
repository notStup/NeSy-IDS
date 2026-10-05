import numpy as np
from itertools import chain
from datetime import datetime
import sys
from io import StringIO

N_CLASSES = 2
ATTACK_CLASS = 1


def validate_rules(
    rule_set,
    x_val,
    y_val_one,
    min_precision=0.85,
    min_support=3,
    shap_allowed_features=None
):

    # Il modello passa tensori TensorFlow; le regole vengono valutate
    # come espressioni Python e richiedono valori scalari numerici.
    if hasattr(x_val, "numpy"):
        x_val = x_val.numpy()
    x_val = np.asarray(x_val)

    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    rules = list(chain.from_iterable(rule_set))

    y_val = np.asarray(y_val_one).astype(int)

    if y_val.ndim == 2:
        if y_val.shape[1] == 1:
            y_val = y_val[:, 0]
        else:
            y_val = np.argmax(y_val, axis=1)

    if y_val.ndim != 1:
        raise ValueError(
            "y_val deve contenere etichette 1D o one-hot 2D"
        )

    # --------------------------------------------------------
    # Manteniamo SOLO le regole ATTACK
    # --------------------------------------------------------

    attack_rules = [
        rule
        for rule in rules
        if int(rule[-1]) == ATTACK_CLASS
    ]

    print("\n=== VALIDAZIONE ATTACK RULES ===")

    print(
        "Regole totali estratte:",
        len(rules)
    )

    print(
        "Regole ATTACK:",
        len(attack_rules)
    )

    print(
        "Regole BENIGN scartate:",
        len(rules) - len(attack_rules)
    )

    

    for i in range(min(3, len(attack_rules))):

        print(f"\nRULE {i+1}:")

        print(attack_rules[i])
    validated_rules = []

    # --------------------------------------------------------
    # Valutazione individuale delle regole
    # --------------------------------------------------------

    def extract_rule_features(rule):
        """
        Estrae gli indici delle feature utilizzate dalla regola.

        Le feature nella struttura delle regole CapsRule sono
        rappresentate come interi/numpy interi.
        L'ultima posizione contiene invece la classe della regola
        e viene esclusa dal chiamante.
        """

        features = []

        for token in rule[:-1]:

            if isinstance(
                token,
                (int, np.integer)
            ):

                feature_index = int(token)

                if 0 <= feature_index < 30:

                    features.append(
                        feature_index
                    )

        return sorted(
            set(features)
        )
    
    for rule_index, rule in enumerate(attack_rules,start=1 ):

        # ----------------------------------------------------
        # SHAP FILTER
        # ----------------------------------------------------

        if shap_allowed_features is not None:

            rule_features = extract_rule_features(
                rule
            )

            # Regola non interpretabile / senza feature
            if not rule_features:
                continue

            # La regola viene mantenuta solo se
            # TUTTE le feature utilizzate sono tra
            # quelle selezionate da SHAP.
            if not all(
                feature in shap_allowed_features
                for feature in rule_features
            ):

                continue

        covered = 0
        true_attack = 0
        false_attack = 0

        for sample_index, x in enumerate(x_val):

            rl = list(rule[:-1])

            i = 3

            while i < (len(rl) - 1):

                try:

                    rl[i] = str(
                        x[int(rl[i])]
                    )

                    i += 6

                except Exception:

                    break

            str_rule = (
                "if ("
                + " ".join(rl)
                + "):\n"
                + "\tprint(True)\n"
                + "else:\n"
                + "\tprint(False)"
            )

            try:

                old_stdout = sys.stdout

                result = StringIO()

                sys.stdout = result

                exec(str_rule)

                sys.stdout = old_stdout

                flg_rule = (
                    result.getvalue()
                    .replace("\n", "")
                )

            except Exception:

                sys.stdout = old_stdout
                continue

            if flg_rule == "True":

                covered += 1

                if y_val[sample_index] == ATTACK_CLASS:

                    true_attack += 1

                else:

                    false_attack += 1

        # ----------------------------------------------------
        # Metriche della singola ATTACK rule
        # ----------------------------------------------------

        if covered == 0:

            precision = 0.0
            recall = 0.0

        else:

            precision = (
                true_attack / covered
            )

            total_attacks = np.sum(
                y_val == ATTACK_CLASS
            )

            recall = (
                true_attack / total_attacks
                if total_attacks > 0
                else 0.0
            )

        print(
            f"Rule {rule_index}: "
            f"covered={covered}, "
            f"TP={true_attack}, "
            f"FP={false_attack}, "
            f"precision={precision:.4f}, "
            f"recall={recall:.4f}"
        )

        # ----------------------------------------------------
        # Promozione a Rule Base
        # ----------------------------------------------------

        if (
            covered >= min_support
            and precision >= min_precision
        ):

            validated_rules.append(rule)

    # --------------------------------------------------------
    # Risultato
    # --------------------------------------------------------

    print(
        "\nRegole ATTACK validate:",
        len(validated_rules)
    )

    return validated_rules
