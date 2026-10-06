import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

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




def get_top_shap_features(importance, top_k):
    """Restituisce gli indici delle feature con importanza SHAP maggiore."""
    importance = np.asarray(importance, dtype=float).reshape(-1)
    top_k = int(top_k)

    if top_k < 0:
        raise ValueError("top_k deve essere >= 0")

    ranking = np.argsort(importance)[::-1]
    return set(int(index) for index in ranking[:top_k])


def show_shap(shap_values, X_explain, selected_features):
    
    shap_values= np.asarray(shap_values)
    X_explain= np.asarray(X_explain)
 
    explanation = shap.Explanation(

        values=shap_values,

        data=X_explain,

        feature_names=selected_features

    )
    shap.plots.beeswarm(explanation, max_display=20)
    shap.plots.bar(explanation, max_display= 20)
    
