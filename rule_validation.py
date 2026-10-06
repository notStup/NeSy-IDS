import numpy as np


N_CLASSES = 2
ATTACK_CLASS = 1


def validate_rules(
    rule_set,
    x_val,
    y_val_one,
    min_precision=0.80,
    min_support=2,
    shap_importance=None,
    return_metadata=False
):
    """
    Valida le ATTACK rules sul validation set.

    Una regola viene mantenuta se:

    - è una ATTACK rule;
    - copre almeno min_support campioni;
    - ha precisione almeno min_precision.

    SHAP non viene usato come filtro binario.
    Viene utilizzato per assegnare un'importanza
    alle feature contenute nella regola e quindi
    un'importanza complessiva alla regola.

    Formato regola:

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
    """

    # ========================================================
    # INPUT
    # ========================================================

    if hasattr(x_val, "numpy"):
        x_val = x_val.numpy()

    x_val = np.asarray(x_val)

    y_val = np.asarray(
        y_val_one
    ).astype(int)

    if y_val.ndim == 2:

        if y_val.shape[1] == 1:

            y_val = y_val[:, 0]

        else:

            y_val = np.argmax(
                y_val,
                axis=1
            )

    if y_val.ndim != 1:

        raise ValueError(
            "y_val deve essere 1D oppure one-hot 2D."
        )

    # ========================================================
    # SHAP IMPORTANCE
    # ========================================================

    if shap_importance is not None:

        shap_importance = np.asarray(
            shap_importance,
            dtype=float
        )

        if shap_importance.ndim != 1:

            raise ValueError(
                "shap_importance deve essere un array 1D."
            )

        shap_total = np.sum(
            shap_importance
        )

        shap_max = np.max(
            shap_importance
        )

    else:

        shap_total = 0.0
        shap_max = 0.0

    # ========================================================
    # NORMALIZZAZIONE RULE SET
    # ========================================================

    rules = []

    def collect_rules(items):

        for item in items:

            if not isinstance(
                item,
                (list, tuple)
            ):

                continue

            # ----------------------------------------------
            # Regola piatta
            # ----------------------------------------------

            if (
                len(item) >= 2
                and isinstance(
                    item[-1],
                    (int, np.integer)
                )
                and int(item[-1]) in (0, 1)
            ):

                rules.append(
                    list(item)
                )

            else:

                collect_rules(
                    item
                )

    collect_rules(
        rule_set
    )

    # ========================================================
    # SOLO ATTACK RULES
    # ========================================================

    attack_rules = [
        rule
        for rule in rules
        if int(rule[-1]) == ATTACK_CLASS
    ]

    print(
        "\n=== VALIDAZIONE ATTACK RULES ==="
    )

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

    # ========================================================
    # ESTRAZIONE FEATURE DELLA REGOLA
    # ========================================================

    def extract_rule_features(rule):

        feature_indices = []

        i = 1

        while i < len(rule) - 1:

            if rule[i] == ")":
                break

            try:

                feature_index = int(
                    rule[i + 4]
                )

            except (
                ValueError,
                TypeError,
                IndexError
            ):

                break

            # ------------------------------------------------
            # Ora non c'è più < 30.
            # Se abbiamo SHAP, usiamo la sua lunghezza.
            # Altrimenti lasciamo passare l'indice.
            # ------------------------------------------------

            if shap_importance is not None:

                if (
                    0
                    <= feature_index
                    < len(shap_importance)
                ):

                    feature_indices.append(
                        feature_index
                    )

            else:

                if feature_index >= 0:

                    feature_indices.append(
                        feature_index
                    )

            # Una condizione occupa 8 elementi:
            #
            # min <= name Feature: index <= max
            #
            # e l'elemento successivo può essere "and"
            # oppure ")".

            i += 8

        return sorted(
            set(feature_indices)
        )

    # ========================================================
    # MATCH REGOLA
    # ========================================================

    def rule_matches(
        rule,
        sample
    ):

        i = 1

        while i < len(rule) - 1:

            if rule[i] == ")":
                break

            try:

                minimum = float(
                    str(rule[i])
                    .strip("[]")
                )

                feature_index = int(
                    rule[i + 4]
                )

                maximum = float(
                    str(rule[i + 6])
                    .strip("[]")
                )

                value = float(
                    np.asarray(
                        sample[
                            feature_index
                        ]
                    ).reshape(-1)[0]
                )

            except (
                ValueError,
                TypeError,
                IndexError
            ):

                return False

            if not (
                minimum
                <= value
                <= maximum
            ):

                return False

            i += 8

        return True

    # ========================================================
    # TOTAL ATTACKS
    # ========================================================

    total_attacks = np.sum(
        y_val == ATTACK_CLASS
    )

    # ========================================================
    # VALIDAZIONE
    # ========================================================

    validated_rules = []

    rule_metadata = []

    support_rejected = 0
    precision_rejected = 0

    for rule_index, rule in enumerate(
        attack_rules,
        start=1
    ):

        # ----------------------------------------------------
        # FEATURE DELLA REGOLA
        # ----------------------------------------------------

        rule_features = (
            extract_rule_features(
                rule
            )
        )

        # ----------------------------------------------------
        # SHAP RULE IMPORTANCE
        # ----------------------------------------------------

        shap_strength = 0.0
        shap_coverage = 0.0

        feature_shap_values = {}

        if (
            shap_importance is not None
            and rule_features
        ):

            valid_features = [

                feature

                for feature in rule_features

                if (
                    0
                    <= feature
                    < len(shap_importance)
                )

            ]

            if valid_features:

                values = np.asarray(

                    [
                        shap_importance[
                            feature
                        ]

                        for feature
                        in valid_features

                    ],

                    dtype=float

                )

                # ------------------------------------------
                # SHAP value per feature
                # ------------------------------------------

                feature_shap_values = {

                    int(feature):
                    float(
                        shap_importance[
                            feature
                        ]
                    )

                    for feature
                    in valid_features

                }

                # ------------------------------------------
                # Forza media delle feature della regola
                #
                # 0 -> feature poco importante
                # 1 -> feature più importante
                # ------------------------------------------

                if shap_max > 0:

                    shap_strength = (
                        np.mean(values)
                        / shap_max
                    )

                # ------------------------------------------
                # Quota della SHAP importance globale
                # rappresentata dalle feature della regola.
                # ------------------------------------------

                if shap_total > 0:

                    shap_coverage = (
                        np.sum(values)
                        / shap_total
                    )

        # ----------------------------------------------------
        # MATCH VALIDATION
        # ----------------------------------------------------

        covered = 0
        true_attack = 0
        false_attack = 0

        for sample_index, sample in enumerate(
            x_val
        ):

            if rule_matches(
                rule,
                sample
            ):

                covered += 1

                if (
                    y_val[sample_index]
                    == ATTACK_CLASS
                ):

                    true_attack += 1

                else:

                    false_attack += 1

        # ----------------------------------------------------
        # PRECISION / RECALL
        # ----------------------------------------------------

        if covered == 0:

            precision = 0.0
            recall = 0.0

        else:

            precision = (
                true_attack
                / covered
            )

            recall = (

                true_attack
                / total_attacks

                if total_attacks > 0

                else 0.0

            )

        # ----------------------------------------------------
        # VALIDAZIONE
        # ----------------------------------------------------

        if covered < min_support:

            support_rejected += 1

            continue

        if precision < min_precision:

            precision_rejected += 1

            continue

        # ----------------------------------------------------
        # RULE PRIORITY
        # ----------------------------------------------------
        #
        # Questo NON decide se la regola è valida.
        #
        # Serve solamente per quantificare la combinazione
        # di affidabilità empirica e importanza SHAP.
        #
        # 0 -> bassa
        # 1 -> alta
        # ----------------------------------------------------

        rule_priority = (
            precision
            * shap_strength
        )

        validated_rules.append(
            rule
        )

        rule_metadata.append({

            "rule_index":
                rule_index,

            "support":
                int(covered),

            "true_attack":
                int(true_attack),

            "false_attack":
                int(false_attack),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "features":
                rule_features,

            "feature_shap_values":
                feature_shap_values,

            "shap_strength":
                float(shap_strength),

            "shap_coverage":
                float(shap_coverage),

            "rule_priority":
                float(rule_priority)

        })

        print(

            f"Rule {rule_index} VALIDATA: "

            f"support={covered}, "

            f"precision={precision:.4f}, "

            f"recall={recall:.4f}, "

            f"SHAP strength={shap_strength:.4f}, "

            f"SHAP coverage={shap_coverage:.4f}, "

            f"priority={rule_priority:.4f}"

        )

    # ========================================================
    # ORDINAMENTO METADATA
    # ========================================================

    rule_metadata.sort(

        key=lambda x:
            x["rule_priority"],

        reverse=True

    )

    # ========================================================
    # RISULTATO
    # ========================================================

    print(
        "\nRegole ATTACK validate:",
        len(validated_rules)
    )

    print(
        "Scartate da supporto:",
        support_rejected,
        "| precisione:",
        precision_rejected
    )

    if return_metadata:

        return (
            validated_rules,
            rule_metadata
        )

    return validated_rules