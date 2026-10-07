import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
from pathlib import Path
class AttackScoreModel(tf.keras.Model):
    """
    Wrapper della FFCN utilizzato da SHAP.

    Input:
        (batch, 30)

    Output:
        (batch, 1) = attack score della capsule ATTACK
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def call(self, inputs, training=False):

        x = tf.cast(
            inputs,
            tf.float32
        )

        # La FFCN originale richiede:
        # (batch, 30, 1)
        x = tf.expand_dims(
            x,
            axis=-1
        )

        capsule_output = self.base_model(
            x,
            training=False
        )

        scores = self.base_model.get_scores(
            capsule_output
        )

        # Solo il punteggio ATTACK
        return scores[:, 1:2]


def compute_attack_shap_importance(
    model,
    X_train,
    y_train,
    background_samples=100,
    explain_samples=500,
    nsamples=200,
    random_state=42
):
    """
    Calcola l'importanza globale delle 30 feature
    rispetto al punteggio ATTACK della FFCN.

    SHAP viene calcolato su campioni ATTACK del training set,
    mentre il background è estratto dal training complessivo.
    """

    # --------------------------------------------------------
    # Conversione input
    # --------------------------------------------------------

    if hasattr(X_train, "numpy"):
        X_train = X_train.numpy()

    if hasattr(y_train, "numpy"):
        y_train = y_train.numpy()

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).reshape(-1)

    # Da (N, 30, 1) a (N, 30)
    if (
        X_train.ndim == 3
        and X_train.shape[-1] == 1
    ):
        X_train = X_train[..., 0]

    if X_train.ndim != 2:
        raise ValueError(
            f"X_train deve avere forma (N, 30), "
            f"ricevuta {X_train.shape}"
        )

    # --------------------------------------------------------
    # Selezione ATTACK
    # --------------------------------------------------------

    attack_mask = (
        y_train == 1
    )

    X_attack = X_train[
        attack_mask
    ]

    if len(X_attack) == 0:
        raise ValueError(
            "Nel training set non sono presenti campioni ATTACK."
        )

    rng = np.random.default_rng(
        random_state
    )

    # --------------------------------------------------------
    # Background misto
    # --------------------------------------------------------

    background_size = min(
        background_samples,
        len(X_train)
    )

    background_indices = rng.choice(
        len(X_train),
        size=background_size,
        replace=False
    )

    background = X_train[
        background_indices
    ].astype(np.float32)

    # --------------------------------------------------------
    # Campioni ATTACK da spiegare
    # --------------------------------------------------------

    explain_size = min(
        explain_samples,
        len(X_attack)
    )

    explain_indices = rng.choice(
        len(X_attack),
        size=explain_size,
        replace=False
    )

    X_explain = X_attack[
        explain_indices
    ].astype(np.float32)

    # --------------------------------------------------------
    # Wrapper FFCN -> attack score
    # --------------------------------------------------------

    attack_model = AttackScoreModel(
        model
    )

    # SHAP's TensorFlow explainer inspects model.inputs. A subclassed
    # Keras model has no `inputs` property, so expose the wrapper through
    # a Functional model with the same (batch, 30) interface.
    shap_input = tf.keras.Input(
        shape=(X_train.shape[1],),
        dtype=tf.float32
    )
    shap_output = attack_model(shap_input, training=False)
    shap_model = tf.keras.Model(
        inputs=shap_input,
        outputs=shap_output
    )

    # --------------------------------------------------------
    # Gradient SHAP
    # --------------------------------------------------------

    explainer = shap.GradientExplainer(
        shap_model,
        background,
        batch_size=50
    )

    shap_values = explainer.shap_values(
        X_explain,
        nsamples=nsamples,
        rseed=random_state
    )

    # --------------------------------------------------------
    # Normalizzazione formato SHAP
    # --------------------------------------------------------

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(
        shap_values
    )

    # Nel caso venga restituito (N, 30, 1)
    if (
        shap_values.ndim == 3
        and shap_values.shape[-1] == 1
    ):
        shap_values = shap_values[..., 0]

    if shap_values.ndim != 2:
        raise ValueError(
            f"Formato SHAP inatteso: "
            f"{shap_values.shape}"
        )

    # --------------------------------------------------------
    # Global feature importance
    # --------------------------------------------------------

    importance = np.mean(
        np.abs(shap_values),
        axis=0
    )

    ranking = np.argsort(
        importance
    )[::-1]

    return (
        importance,
        ranking
    )

def compute_attack_shap_importance(
    model,
    X_train,
    y_train,
    background_samples=100,
    explain_samples=500,
    nsamples=200,
    random_state=42
):
    """
    Calcola l'importanza globale delle 30 feature
    rispetto al punteggio ATTACK della FFCN.

    SHAP viene calcolato su campioni ATTACK del training set,
    mentre il background è estratto dal training complessivo.
    """

    # --------------------------------------------------------
    # Conversione input
    # --------------------------------------------------------

    if hasattr(X_train, "numpy"):
        X_train = X_train.numpy()

    if hasattr(y_train, "numpy"):
        y_train = y_train.numpy()

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).reshape(-1)

    # Da (N, 30, 1) a (N, 30)
    if (
        X_train.ndim == 3
        and X_train.shape[-1] == 1
    ):
        X_train = X_train[..., 0]

    if X_train.ndim != 2:
        raise ValueError(
            f"X_train deve avere forma (N, 30), "
            f"ricevuta {X_train.shape}"
        )

    # --------------------------------------------------------
    # Selezione ATTACK
    # --------------------------------------------------------

    attack_mask = (
        y_train == 1
    )

    X_attack = X_train[
        attack_mask
    ]

    if len(X_attack) == 0:
        raise ValueError(
            "Nel training set non sono presenti campioni ATTACK."
        )

    rng = np.random.default_rng(
        random_state
    )

    # --------------------------------------------------------
    # Background misto
    # --------------------------------------------------------

    background_size = min(
        background_samples,
        len(X_train)
    )

    background_indices = rng.choice(
        len(X_train),
        size=background_size,
        replace=False
    )

    background = X_train[
        background_indices
    ].astype(np.float32)

    # --------------------------------------------------------
    # Campioni ATTACK da spiegare
    # --------------------------------------------------------

    explain_size = min(
        explain_samples,
        len(X_attack)
    )

    explain_indices = rng.choice(
        len(X_attack),
        size=explain_size,
        replace=False
    )

    X_explain = X_attack[
        explain_indices
    ].astype(np.float32)

    # --------------------------------------------------------
    # Wrapper FFCN -> attack score
    # --------------------------------------------------------

    attack_model = AttackScoreModel(
        model
    )

    shap_input = tf.keras.Input(
        shape=(X_train.shape[1],),
        dtype=tf.float32
    )

    shap_output = attack_model(
        shap_input,
        training=False
    )

    shap_model = tf.keras.Model(
        inputs=shap_input,
        outputs=shap_output
    )

    # --------------------------------------------------------
    # Gradient SHAP
    # --------------------------------------------------------

    explainer = shap.GradientExplainer(
        shap_model,
        background,
        batch_size=50
    )

    shap_values = explainer.shap_values(
        X_explain,
        nsamples=nsamples,
        rseed=random_state
    )

    # --------------------------------------------------------
    # Normalizzazione formato SHAP
    # --------------------------------------------------------

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(
        shap_values
    )

    # Nel caso venga restituito (N, 30, 1)
    if (
        shap_values.ndim == 3
        and shap_values.shape[-1] == 1
    ):
        shap_values = shap_values[..., 0]

    if shap_values.ndim != 2:
        raise ValueError(
            f"Formato SHAP inatteso: "
            f"{shap_values.shape}"
        )

    # --------------------------------------------------------
    # Global feature importance
    # --------------------------------------------------------

    importance = np.mean(
        np.abs(shap_values),
        axis=0
    )

    ranking = np.argsort(
        importance
    )[::-1]

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return (
        importance,
        ranking,
        shap_values,
        X_explain
    )




def get_top_shap_features(importance, top_k=20):
    """Return the indices of the ``top_k`` most important features."""
    importance = np.asarray(importance).reshape(-1)
    if importance.size == 0:
        return set()

    top_k = max(0, min(int(top_k), importance.size))
    ranking = np.argsort(importance)[::-1]
    return set(int(index) for index in ranking[:top_k])


def show_shap(
    shap_values,
    X_explain,
    selected_features,
    output_dir="."
):

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    shap_values = np.asarray(shap_values)
    X_explain = np.asarray(X_explain)

    explanation = shap.Explanation(
        values=shap_values,
        data=X_explain,
        feature_names=selected_features
    )

    # BEESWARM
    shap.plots.beeswarm(
        explanation,
        max_display=20,
        show=False
    )

    fig = plt.gcf()

    beeswarm_paths = (
        output_dir / "shap_beeswarm.png",
        output_dir / "shap_beeswarm.pdf"
    )

    for path in beeswarm_paths:
        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.close(fig)


    # ========================================================
    # BAR
    # ========================================================

    shap.plots.bar(
        explanation,
        max_display=20,
        show=False
    )

    fig = plt.gcf()

    bar_paths = (
        output_dir / "shap_barplot.png",
        output_dir / "shap_barplot.pdf"
    )

    for path in bar_paths:
        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.close(fig)


    print(
        f"\n[SHAP] Beeswarm salvato: "
        f"{beeswarm_paths[0].resolve()} e "
        f"{beeswarm_paths[1].resolve()}"
    )

    print(
        f"[SHAP] Bar plot salvato: "
        f"{bar_paths[0].resolve()} e "
        f"{bar_paths[1].resolve()}"
    )




def logic_shap(X_explain,
    shap_values,
    rule_set):
    results = []
    for rule in rule_set:
        feature_index = int(rule[5])
        lower = float(rule[1])
        upper = float(rule[7])

        feature_values = np.asarray(X_explain)[:, feature_index]
        feature_shap = np.asarray(shap_values)[:, feature_index]

        mask = (
            (feature_values <= upper) &
            (feature_values >= lower)
        )

        selected_shap = feature_shap[mask]
        n_samples = len(selected_shap)

        if n_samples == 0:
            result = {
                "feature_index": feature_index,
                "lower": lower,
                "upper": upper,
                "n_samples": 0,
                "mean_shap": 0.0,
                "mean_abs_shap": 0.0,
                "positive_ratio": 0.0,
                "shap_score": 0.0,
            }
        else:
            mean_shap = float(np.mean(selected_shap))
            mean_abs_shap = float(np.mean(np.abs(selected_shap)))
            positive_ratio = float(np.mean(selected_shap > 0))

            result = {
                "feature_index": feature_index,
                "lower": lower,
                "upper": upper,
                "n_samples": n_samples,
                "mean_shap": mean_shap,
                "mean_abs_shap": mean_abs_shap,
                "positive_ratio": positive_ratio,
                "shap_score": mean_shap,
            }
        results.append(result)

    return results
        

   
   
def print_shap_rules(rule_set, shap_results, feature_names):

    for i, (rule, result) in enumerate(zip(rule_set, shap_results), start=1):

        feature_index = result["feature_index"]
        feature_name = feature_names[feature_index]

        print("\n" + "=" * 70)
        print(f"RULE {i}")
        print("=" * 70)

        print(f"Feature       : {feature_name}")
        print(f"Feature index : {feature_index}")
        print(f"Range         : [{result['lower']}, {result['upper']}]")
        print(f"Class         : {rule[9]}")

        print(f"Samples       : {result['n_samples']}")
        print(f"Mean SHAP     : {result['mean_shap']:+.6f}")
        print(f"Mean |SHAP|   : {result['mean_abs_shap']:.6f}")
        print(f"Positive ratio: {result['positive_ratio']:.2%}")
