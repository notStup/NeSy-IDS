import numpy as np
from squash import squash_arr


# ============================================================
# CONFIGURAZIONE
# ============================================================

N_CLASSES = 2
# 0 = BENIGN
# 1 = ATTACK


# ============================================================
# FIND MIN / MAX COEFFICIENT
# ============================================================

def find_min_max_coef(
    i,
    j,
    cls,
    couple_slice,
    pred_vect,
    out_vect
):
    """
    Individua le capsule piu rilevanti per una determinata
    capsule/classificazione.

    Parametri
    ---------
    i : int
        Indice del campione nel batch.

    j : int
        Indice del layer.

    cls : int
        Classe/capsula di output considerata.

    couple_slice : np.ndarray
        Slice dei coupling coefficients tra il layer corrente
        e il layer precedente.

    pred_vect : list[np.ndarray]
        Vettori di predizione delle capsule.

    out_vect : list[np.ndarray]
        Output delle capsule.

    Returns
    -------
    np.ndarray
        Indici delle capsule rilevanti.
    """

    # Ordina i coupling coefficients in ordine decrescente.
    ordered_indices = np.argsort(
        couple_slice
    )[::-1]

    gamma = 0.9

    # Soglia basata sull'output della capsule corrente.
    threshold = (
        gamma
        * np.linalg.norm(
            out_vect[j][i, cls]
        )
    )

    s = 0
    contribution_norm = 0
    k = 0

    # Seleziona progressivamente le capsule con coupling
    # maggiore finche il contributo supera la soglia.
    while (
        contribution_norm < threshold
        and k < ordered_indices.size
    ):

        capsule_index = ordered_indices[k]

        s += (
            couple_slice[capsule_index]
            * pred_vect[j][i, capsule_index, cls]
        )

        contribution_norm = np.linalg.norm(
            squash_arr(s)
        )

        k += 1

    return ordered_indices[:k]


# ============================================================
# RECURSIVE COUPLING COEFFICIENT EXTRACTION
# ============================================================

def recursive_coupl_coeff(
    i,
    j,
    cls,
    couple_slice,
    coupl_coeff,
    pred_vect,
    out_vect
):
    """
    Risale ricorsivamente attraverso i layer della rete per
    individuare le capsule rilevanti fino al layer di input.
    """

    if j == 0:

        # Siamo arrivati al layer di input.
        return find_min_max_coef(
            i,
            j,
            cls,
            couple_slice,
            pred_vect,
            out_vect
        )

    # Individua le capsule piu rilevanti del layer corrente.
    max_nodes = find_min_max_coef(
        i,
        j,
        cls,
        couple_slice,
        pred_vect,
        out_vect
    )

    relevant_nodes = []

    # Per ogni capsule rilevante risaliamo al layer precedente.
    for capsule_index in max_nodes:

        previous_coupling_slice = (
            coupl_coeff[j - 1][
                i,
                :,
                capsule_index
            ]
        )

        relevant_nodes.append(
            recursive_coupl_coeff(
                i,
                j - 1,
                capsule_index,
                previous_coupling_slice,
                coupl_coeff,
                pred_vect,
                out_vect
            )
        )

    return relevant_nodes


# ============================================================
# FLATTEN DELLE STRUTTURE RICORSIVE
# ============================================================

output = []


def reemovNestings(value):
    """
    Appiattisce ricorsivamente una struttura composta da
    liste e ndarray.

    Gli elementi finali vengono raccolti nella lista globale
    'output'.
    """

    for item in value:

        if (
            isinstance(item, np.ndarray)
            or isinstance(item, list)
        ):
            reemovNestings(item)

        else:
            output.append(item)


# ============================================================
# CAPSRULE - ESTRAZIONE DELLE REGOLE
# ============================================================

def extract_rules_boundary(
    input_dt,
    coupl_coeff,
    pred_vect,
    out_vect,
    pred,
    features,
    rule_arr_class= None
):
    """
    Estrae le regole CapsRule a partire dai dati prodotti
    dalla rete FFCN.

    Parametri
    ---------
    input_dt : np.ndarray
        Batch di input.

    coupl_coeff : list[np.ndarray]
        Coupling coefficients dei layer.

    pred_vect : list[np.ndarray]
        Vettori di predizione delle capsule.

    out_vect : list[np.ndarray]
        Output delle capsule.

    pred : np.ndarray
        Score delle classi prodotti dal modello.

    Returns
    -------
    list[list[dict]]
        Accumulatore delle regole aggregate per classe. Ogni regola
        mappa gli indici delle feature ai valori osservati nei batch.
    """

    global output

    # Ogni estrazione deve partire da uno stato vuoto.
    output = []

    # Classe predetta per ogni campione.
    predicted_classes = np.argmax(
        pred,
        axis=1
    )

    # Coupling coefficients dell'ultimo layer.
    couple_slice = coupl_coeff[-1]
    if rule_arr_class is None:
    # Un insieme di regole per ogni classe.
        rule_arr_class = [
            []
            for _ in range(N_CLASSES)
        ]

    # --------------------------------------------------------
    # Estrazione per ogni campione del batch
    # --------------------------------------------------------

    for i in range(len(predicted_classes)):

        # Classe predetta dal modello.
        current_class = predicted_classes[i]

        # Coupling coefficients del campione i
        # verso la classe predetta.
        arr_slice = couple_slice[
            i,
            :,
            current_class
        ]

        # Partiamo dall'ultimo layer e risaliamo
        # ricorsivamente fino all'input.
        last_layer = len(coupl_coeff) - 1

        relevant_features = recursive_coupl_coeff(
            i,
            last_layer,
            current_class,
            arr_slice,
            coupl_coeff,
            pred_vect,
            out_vect
        )

        # ----------------------------------------------------
        # Flatten delle feature rilevanti
        # ----------------------------------------------------

        flattened_rules = []

        for sublist in relevant_features:

            reemovNestings(sublist)

            # Elimina duplicati e ordina gli indici.
            unique_features = list(set(output))
            unique_features.sort()

            flattened_rules.append(unique_features)

            # Reset per il prossimo ramo.
            output = []

        # ----------------------------------------------------
        # Costruzione delle condizioni
        # ----------------------------------------------------

        for feature_indices in flattened_rules:

            rule_dict = {}

            found_existing_rule = False
            existing_rule_index = None
            redundant_conditions = []

            # ------------------------------------------------
            # Verifica se una struttura simile esiste gia
            # per la stessa classe.
            # ------------------------------------------------

            try:

                for existing_rule in rule_arr_class[current_class]:

                    existing_features = list(existing_rule)

                    if (
                        set(feature_indices).issubset(
                            existing_features
                        )
                        or
                        set(existing_features).issubset(
                            feature_indices
                        )
                    ):

                        existing_rule_index = (
                            rule_arr_class[current_class].index(
                                existing_rule
                            )
                        )

                        found_existing_rule = True

                        # Legge di assorbimento:
                        # a AND (a OR b) = a
                        if len(feature_indices) > len(existing_features):
                            feature_indices = existing_features
                        else:
                            redundant_conditions = list(
                                set(existing_features)
                                - set(feature_indices)
                            )

                        break

            except Exception as exc:

                print(
                    f"Errore durante il confronto delle regole: {exc}"
                )

            # ------------------------------------------------
            # Inserimento delle feature nella regola
            # ------------------------------------------------

            for feature_index in feature_indices:

                if found_existing_rule:

                    rule_arr_class[
                        current_class
                    ][
                        existing_rule_index
                    ][
                        feature_index
                    ].append(
                        input_dt[
                            i,
                            feature_index
                        ].item()
                    )

                else:

                    rule_dict.setdefault(
                        feature_index,
                        []
                    ).append(
                        input_dt[
                            i,
                            feature_index
                        ].item()
                    )

            # ------------------------------------------------
            # Rimozione di eventuali condizioni ridondanti
            # ------------------------------------------------

            if (
                found_existing_rule
                and len(redundant_conditions) > 0
            ):

                for feature_index in redundant_conditions:

                    del rule_arr_class[
                        current_class
                    ][
                        existing_rule_index
                    ][
                        feature_index
                    ]

            # Se non esisteva ancora, aggiunge la nuova regola.
            if not found_existing_rule:

                rule_arr_class[
                    current_class
                ].append(rule_dict)

    # Restituisce l'accumulatore strutturato. La conversione in
    # espressioni serializzate avviene una sola volta, dopo tutti i batch.
    return rule_arr_class
