import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

import numpy as np
import joblib

from preprocessing import preprocess_dataset
from squash import squash, safe_norm
from rule_extraction import extract_rules_boundary
from rule_validation import validate_rules
from rule_evaluation import evaluate_rules_boundary
from shap_guidance import (
    compute_attack_shap_importance,
    get_top_shap_features,
)

# ============================================================
# ABLATION STUDY: SHAP vs NO-SHAP
# ============================================================
#
# IMPORTANT:
# This script trains the FFCN ONLY ONCE.
# It then extracts ONE candidate Rule Set and evaluates it
# twice:
#
#   A) without SHAP filtering
#   B) with SHAP filtering
#
# Therefore both experiments use:
#   - the same FFCN
#   - the same best weights
#   - the same train/validation/test split
#   - the same extracted candidate rules
#   - the same validation set
#   - the same test set
#
# The only difference is the SHAP feature filter applied
# during rule validation.
#
# This file is intentionally separate from the main experiment.
# ============================================================


# ============================================================
# CONFIGURAZIONE
# ============================================================

BATCH_SIZE = 100

N_CLASSES = 2
# 0 = BENIGN
# 1 = ATTACK

SIZE_LAYERS = [25, 20, 15, 8, N_CLASSES]
CAPS_DIMS = [30, 25, 20, 15, 10]

ROUTING_ITERATIONS = 3
INPUT_SIZE = 30

LEARNING_RATE = 0.001
EPOCHS = 10

RECONSTRUCTION_COEFFICIENT = 5.0e-15

# Esperimento ridotto locale.
TRAIN_FRACTION = 0.005

# Numero di flow utilizzati per la valutazione finale.
TEST_SAMPLES = 5000

# Percentuale del training ridotto destinata alla validation.
VALIDATION_FRACTION = 0.10

RANDOM_STATE = 42


# ============================================================
# SHAP CONFIGURATION
# ============================================================

SHAP_BACKGROUND_SAMPLES = 100
SHAP_EXPLAIN_SAMPLES = 500
SHAP_NSAMPLES = 200
SHAP_TOP_K = 20


# ============================================================
# RIPRODUCIBILITÀ
# ============================================================

tf.keras.utils.set_random_seed(RANDOM_STATE)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


# ============================================================
# DATASET
# ============================================================

X_train, X_test, y_train, y_test, selected_features = (
    preprocess_dataset()
)

print("Dataset preprocessato:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# ============================================================
# ESPERIMENTO RIDOTTO
# ============================================================

X_train, _, y_train, _ = train_test_split(
    X_train,
    y_train,
    train_size=TRAIN_FRACTION,
    random_state=RANDOM_STATE,
    stratify=y_train,
)

print("\nTraining ridotto:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)


# ============================================================
# TRAIN / VALIDATION
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=VALIDATION_FRACTION,
    random_state=RANDOM_STATE,
    stratify=y_train,
)

print("\nSuddivisione train / validation:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("y_train:", y_train.shape)
print("y_val:", y_val.shape)


# ============================================================
# TEST RIDOTTO
# ============================================================

X_test, _, y_test, _ = train_test_split(
    X_test,
    y_test,
    train_size=TEST_SAMPLES,
    random_state=RANDOM_STATE,
    stratify=y_test,
)

print("\nTest ridotto:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# SALVATAGGIO SPLIT CONGELATO
# ============================================================
#
# Utile per documentare l'ablation study.
# Non viene usato per cambiare il flusso: gli array già presenti
# in memoria restano quelli usati da entrambi i rami.
# ============================================================

experiments_dir = Path("Experiments")
experiments_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "random_state": RANDOM_STATE,
        "train_fraction": TRAIN_FRACTION,
        "validation_fraction": VALIDATION_FRACTION,
        "test_samples": TEST_SAMPLES,
    },
    experiments_dir / "ablation_frozen_split.joblib",
)


# ============================================================
# CONVERSIONE TENSORFLOW
# ============================================================

X_train = tf.convert_to_tensor(
    X_train.to_numpy(),
    dtype=tf.float32,
)

X_val = tf.convert_to_tensor(
    X_val.to_numpy(),
    dtype=tf.float32,
)

X_test = tf.convert_to_tensor(
    X_test.to_numpy(),
    dtype=tf.float32,
)

y_train = tf.convert_to_tensor(
    y_train.to_numpy().reshape(-1),
    dtype=tf.int32,
)

y_val = tf.convert_to_tensor(
    y_val.to_numpy().reshape(-1),
    dtype=tf.int32,
)

y_test = tf.convert_to_tensor(
    y_test.to_numpy().reshape(-1),
    dtype=tf.int32,
)


# ============================================================
# INPUT CAPSULE
# ============================================================

# (batch, 30) -> (batch, 30, 1)

X_train = tf.expand_dims(
    X_train,
    axis=-1,
)

X_val = tf.expand_dims(
    X_val,
    axis=-1,
)

X_test = tf.expand_dims(
    X_test,
    axis=-1,
)

print("\nInput capsule:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)


# ============================================================
# CAPSULE LAYER
# ============================================================

class CapsuleLayer(layers.Layer):

    def __init__(
        self,
        num_capsules,
        capsule_dim,
        routing_iterations=3,
    ):
        super().__init__()

        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.routing_iterations = routing_iterations

        # Informazioni richieste da CapsRule.
        self.last_coupling_coefficients = None
        self.last_caps_predicted = None
        self.last_caps_output = None

    def build(self, input_shape):

        self.input_capsules = int(input_shape[1])
        self.input_dim = int(input_shape[2])

        self.W = self.add_weight(
            name="weights",
            shape=(
                1,
                self.input_capsules,
                self.num_capsules,
                self.capsule_dim,
                self.input_dim,
            ),
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=1.0,
            ),
            trainable=True,
        )

    def call(self, inputs, training=False):

        # ----------------------------------------------------
        # Prediction vectors
        # ----------------------------------------------------

        inputs64 = tf.cast(
            inputs,
            tf.float64,
        )

        W64 = tf.cast(
            tf.squeeze(
                self.W,
                axis=0,
            ),
            tf.float64,
        )

        caps_predicted = tf.einsum(
            "bid,ijod->bijo",
            inputs64,
            W64,
        )

        # ----------------------------------------------------
        # Routing weights iniziali
        # ----------------------------------------------------

        batch_size = tf.shape(inputs64)[0]

        raw_weights = tf.zeros(
            (
                batch_size,
                self.input_capsules,
                self.num_capsules,
            ),
            dtype=tf.float64,
        )

        coupling_coefficients = None
        caps_output = None

        # ----------------------------------------------------
        # Dynamic routing
        # ----------------------------------------------------

        for _ in range(self.routing_iterations):

            coupling_coefficients = tf.nn.softmax(
                raw_weights,
                axis=-1,
            )

            weighted_predictions = (
                coupling_coefficients[..., tf.newaxis]
                * caps_predicted
            )

            weighted_sum = tf.reduce_sum(
                weighted_predictions,
                axis=1,
            )

            caps_output = squash(
                weighted_sum,
                axis=-1,
            )

            agreement = tf.reduce_sum(
                caps_predicted
                * caps_output[:, tf.newaxis, :, :],
                axis=-1,
            )

            raw_weights = (
                raw_weights + agreement
            )

        # ----------------------------------------------------
        # COUPLING COEFFICIENTS FINALI
        # ----------------------------------------------------

        coupling_coefficients = tf.nn.softmax(
            raw_weights,
            axis=-1,
        )

        # ----------------------------------------------------
        # Salvataggio dati per CapsRule
        # ----------------------------------------------------

        self.last_coupling_coefficients = (
            coupling_coefficients
        )

        self.last_caps_predicted = (
            caps_predicted
        )

        self.last_caps_output = (
            caps_output
        )

        return tf.cast(
            caps_output,
            inputs.dtype,
        )


# ============================================================
# FFCN
# ============================================================

class FFCN(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.capsule_layers = [

            CapsuleLayer(
                num_capsules=SIZE_LAYERS[0],
                capsule_dim=CAPS_DIMS[0],
                routing_iterations=ROUTING_ITERATIONS,
            ),

            CapsuleLayer(
                num_capsules=SIZE_LAYERS[1],
                capsule_dim=CAPS_DIMS[1],
                routing_iterations=ROUTING_ITERATIONS,
            ),

            CapsuleLayer(
                num_capsules=SIZE_LAYERS[2],
                capsule_dim=CAPS_DIMS[2],
                routing_iterations=ROUTING_ITERATIONS,
            ),

            CapsuleLayer(
                num_capsules=SIZE_LAYERS[3],
                capsule_dim=CAPS_DIMS[3],
                routing_iterations=ROUTING_ITERATIONS,
            ),

            CapsuleLayer(
                num_capsules=SIZE_LAYERS[4],
                capsule_dim=CAPS_DIMS[4],
                routing_iterations=ROUTING_ITERATIONS,
            ),
        ]

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.decoder = tf.keras.Sequential([

            layers.Dense(
                5,
                activation="relu",
            ),

            layers.Dense(
                15,
                activation="relu",
            ),

            layers.Dense(
                20,
                activation="relu",
            ),

            layers.Dense(
                INPUT_SIZE,
                activation="sigmoid",
            ),
        ])

        # ----------------------------------------------------
        # Metriche
        # ----------------------------------------------------

        self.loss_tracker = tf.keras.metrics.Mean(
            name="loss"
        )

        self.margin_loss_tracker = tf.keras.metrics.Mean(
            name="margin_loss"
        )

        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )

        self.accuracy_tracker = tf.keras.metrics.Mean(
            name="accuracy"
        )

    @property
    def metrics(self):

        return [
            self.loss_tracker,
            self.margin_loss_tracker,
            self.reconstruction_loss_tracker,
            self.accuracy_tracker,
        ]

    def call(self, inputs, training=False):

        x = inputs

        for capsule_layer in self.capsule_layers:

            x = capsule_layer(
                x,
                training=training,
            )

        return x

    def get_scores(self, capsule_output):

        return safe_norm(
            capsule_output,
            axis=-1,
        )

    def get_predictions(self, capsule_output):

        scores = self.get_scores(
            capsule_output
        )

        return tf.argmax(
            scores,
            axis=-1,
            output_type=tf.int32,
        )

    def reconstruct(
        self,
        capsule_output,
        y_true,
    ):

        y_one_hot = tf.one_hot(
            y_true,
            depth=N_CLASSES,
            dtype=capsule_output.dtype,
        )

        mask = y_one_hot[..., tf.newaxis]

        masked_capsules = (
            capsule_output * mask
        )

        decoder_input = tf.reshape(
            masked_capsules,
            (
                -1,
                N_CLASSES * CAPS_DIMS[-1],
            ),
        )

        return self.decoder(
            decoder_input
        )

    def compute_losses(
        self,
        inputs,
        y_true,
        capsule_output,
    ):

        margin = margin_loss(
            y_true,
            capsule_output,
        )

        reconstruction_output = self.reconstruct(
            capsule_output,
            y_true,
        )

        input_flat = tf.reshape(
            inputs,
            (
                tf.shape(inputs)[0],
                INPUT_SIZE,
            ),
        )

        reconstruction_loss = tf.reduce_mean(
            tf.square(
                input_flat
                - reconstruction_output
            )
        )

        total_loss = (
            margin
            + RECONSTRUCTION_COEFFICIENT
            * reconstruction_loss
        )

        return (
            total_loss,
            margin,
            reconstruction_loss,
        )

    def train_step(self, data):

        inputs, y_true = data

        y_true = tf.cast(
            tf.reshape(
                y_true,
                [-1],
            ),
            tf.int32,
        )

        with tf.GradientTape() as tape:

            capsule_output = self(
                inputs,
                training=True,
            )

            (
                total_loss,
                margin,
                reconstruction,
            ) = self.compute_losses(
                inputs,
                y_true,
                capsule_output,
            )

        gradients = tape.gradient(
            total_loss,
            self.trainable_variables,
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.trainable_variables,
            )
        )

        predictions = self.get_predictions(
            capsule_output
        )

        accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    predictions,
                    y_true,
                ),
                tf.float32,
            )
        )

        self.loss_tracker.update_state(
            total_loss
        )

        self.margin_loss_tracker.update_state(
            margin
        )

        self.reconstruction_loss_tracker.update_state(
            reconstruction
        )

        self.accuracy_tracker.update_state(
            accuracy
        )

        return {
            "loss":
                self.loss_tracker.result(),
            "margin_loss":
                self.margin_loss_tracker.result(),
            "reconstruction_loss":
                self.reconstruction_loss_tracker.result(),
            "accuracy":
                self.accuracy_tracker.result(),
        }

    def test_step(self, data):

        inputs, y_true = data

        y_true = tf.cast(
            tf.reshape(
                y_true,
                [-1],
            ),
            tf.int32,
        )

        capsule_output = self(
            inputs,
            training=False,
        )

        (
            total_loss,
            margin,
            reconstruction,
        ) = self.compute_losses(
            inputs,
            y_true,
            capsule_output,
        )

        predictions = self.get_predictions(
            capsule_output
        )

        accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    predictions,
                    y_true,
                ),
                tf.float32,
            )
        )

        self.loss_tracker.update_state(
            total_loss
        )

        self.margin_loss_tracker.update_state(
            margin
        )

        self.reconstruction_loss_tracker.update_state(
            reconstruction
        )

        self.accuracy_tracker.update_state(
            accuracy
        )

        return {
            "loss":
                self.loss_tracker.result(),
            "margin_loss":
                self.margin_loss_tracker.result(),
            "reconstruction_loss":
                self.reconstruction_loss_tracker.result(),
            "accuracy":
                self.accuracy_tracker.result(),
        }


# ============================================================
# MARGIN LOSS
# ============================================================

def margin_loss(
    y_true,
    y_pred,
):

    capsule_lengths = safe_norm(
        y_pred,
        axis=-1,
    )

    y_true = tf.one_hot(
        tf.cast(
            y_true,
            tf.int32,
        ),
        depth=N_CLASSES,
        dtype=capsule_lengths.dtype,
    )

    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5

    positive_loss = (
        y_true
        * tf.square(
            tf.maximum(
                0.0,
                m_plus - capsule_lengths,
            )
        )
    )

    negative_loss = (
        lambda_
        * (1.0 - y_true)
        * tf.square(
            tf.maximum(
                0.0,
                capsule_lengths - m_minus,
            )
        )
    )

    return tf.reduce_mean(
        tf.reduce_sum(
            positive_loss + negative_loss,
            axis=-1,
        )
    )


# ============================================================
# CREAZIONE MODELLO
# ============================================================

model = FFCN()

dummy_output = model(
    X_train[:BATCH_SIZE],
    training=False,
)

print("\nOutput FFCN:", dummy_output.shape)


# ============================================================
# OPTIMIZER
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        global_clipnorm=1.0,
    )
)


# ============================================================
# CHECKPOINT
# ============================================================

models_dir = Path("models")
models_dir.mkdir(
    parents=True,
    exist_ok=True,
)

checkpoint_path = (
    models_dir / "ablation_best_ffcn.weights.h5"
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(checkpoint_path),
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=10,
    restore_best_weights=True,
    verbose=1,
)


# ============================================================
# TRAINING FFCN - UNA SOLA VOLTA
# ============================================================

print("\n============================================================")
print("=== TRAINING FFCN - UNA SOLA VOLTA PER ABLATION STUDY ===")
print("============================================================")

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_val,
        y_val,
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        checkpoint,
        early_stopping,
    ],
)


# ============================================================
# CARICAMENTO BEST MODEL
# ============================================================

model.load_weights(
    str(checkpoint_path)
)

print("\n=== FFCN BEST MODEL CARICATO ===")
print("Checkpoint:", checkpoint_path.resolve())


# ============================================================
# PREDIZIONI FFCN
# ============================================================

print("\n=== DIAGNOSTICA FFCN ===")

val_output = model(
    X_val,
    training=False,
)

val_predictions = (
    model.get_predictions(
        val_output
    )
    .numpy()
)

test_output = model(
    X_test,
    training=False,
)

test_predictions = (
    model.get_predictions(
        test_output
    )
    .numpy()
)


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

print("\nVALIDATION")

print(
    "Ground truth BENIGN:",
    int(np.sum(y_val.numpy() == 0))
)

print(
    "Ground truth ATTACK:",
    int(np.sum(y_val.numpy() == 1))
)

print(
    "Predicted BENIGN:",
    int(np.sum(val_predictions == 0))
)

print(
    "Predicted ATTACK:",
    int(np.sum(val_predictions == 1))
)


# ------------------------------------------------------------
# Test
# ------------------------------------------------------------

print("\nTEST")

print(
    "Ground truth BENIGN:",
    int(np.sum(y_test.numpy() == 0))
)

print(
    "Ground truth ATTACK:",
    int(np.sum(y_test.numpy() == 1))
)

print(
    "Predicted BENIGN:",
    int(np.sum(test_predictions == 0))
)

print(
    "Predicted ATTACK:",
    int(np.sum(test_predictions == 1))
)


# ============================================================
# EVALUATION FFCN SUL TEST
# ============================================================

print("\n=== TEST FFCN ===")

ffcn_test_results = model.evaluate(
    X_test,
    y_test,
    batch_size=BATCH_SIZE,
    return_dict=True,
)

print("\nRisultati FFCN:")

for name, value in ffcn_test_results.items():

    print(
        f"{name}: {value:.6f}"
    )


# ============================================================
# METRICHE FFCN - CLASSE ATTACK
# ============================================================

y_test_np = y_test.numpy()

ffcn_precision = precision_score(
    y_test_np,
    test_predictions,
    pos_label=1,
    zero_division=0,
)

ffcn_recall = recall_score(
    y_test_np,
    test_predictions,
    pos_label=1,
    zero_division=0,
)

ffcn_f1 = f1_score(
    y_test_np,
    test_predictions,
    pos_label=1,
    zero_division=0,
)

print("\n=== FFCN ATTACK METRICS ===")

print(
    f"Precision: {ffcn_precision:.4f}"
)

print(
    f"Recall: {ffcn_recall:.4f}"
)

print(
    f"F1: {ffcn_f1:.4f}"
)


# ============================================================
# SHAP GLOBAL FEATURE IMPORTANCE
# ============================================================
#
# SHAP viene calcolato una sola volta sullo stesso FFCN che verrà
# usato per estrarre il Rule Set.
#
# IMPORTANTE:
# il ramo NO-SHAP NON viene ottenuto riaddestrando il modello.
# ============================================================

print(
    "\n=== SHAP FEATURE IMPORTANCE ==="
)

(
    shap_importance,
    shap_ranking,
) = compute_attack_shap_importance(
    model,
    X_train,
    y_train,
    background_samples=SHAP_BACKGROUND_SAMPLES,
    explain_samples=SHAP_EXPLAIN_SAMPLES,
    nsamples=SHAP_NSAMPLES,
    random_state=RANDOM_STATE,
)

shap_allowed_features = (
    get_top_shap_features(
        shap_importance,
        SHAP_TOP_K,
    )
)

print("\nTop feature SHAP:")

for rank, feature_index in enumerate(
    shap_ranking[:SHAP_TOP_K],
    start=1,
):

    feature_name = (
        selected_features[feature_index]
        if feature_index < len(selected_features)
        else f"Feature {feature_index}"
    )

    print(
        f"{rank}. "
        f"Feature {feature_index} "
        f"({feature_name}) "
        f"importance="
        f"{shap_importance[feature_index]:.6f}"
    )

print(
    "\nFeature utilizzabili dalle ATTACK rules:",
    sorted(shap_allowed_features),
)


# ============================================================
# SALVATAGGIO SHAP
# ============================================================

shap_results = {
    "importance": shap_importance,
    "ranking": shap_ranking,
    "top_k": SHAP_TOP_K,
    "allowed_features": sorted(
        shap_allowed_features
    ),
}

joblib.dump(
    shap_results,
    experiments_dir / "ablation_shap_feature_importance.joblib",
)


# ============================================================
# ESTRAZIONE CAPSRule
# ============================================================
#
# QUESTA PARTE VIENE ESEGUITA UNA SOLA VOLTA.
#
# Il Rule Set ottenuto qui è il Rule Set candidato comune
# ai due esperimenti:
#
#   NO SHAP
#   SHAP
# ============================================================

print("\n============================================================")
print("=== ESTRAZIONE CAPSRule - UNA SOLA VOLTA ===")
print("============================================================")

rule_set = []

num_training_batches = (
    (len(X_train) + BATCH_SIZE - 1)
    // BATCH_SIZE
)

print(
    "Batch di training da elaborare:",
    num_training_batches,
)


for batch_number, start in enumerate(
    range(
        0,
        len(X_train),
        BATCH_SIZE,
    ),
    start=1,
):

    end = min(
        start + BATCH_SIZE,
        len(X_train),
    )

    X_batch = X_train[start:end]

    # Forward pass della FFCN già addestrata.
    # NON modifica i pesi.
    capsule_output = model(
        X_batch,
        training=False,
    )

    # --------------------------------------------------------
    # Dati necessari a CapsRule
    # --------------------------------------------------------

    coupl_coeff = [
        layer.last_coupling_coefficients.numpy()
        for layer in model.capsule_layers
    ]

    pred_vect = [
        layer.last_caps_predicted.numpy()
        for layer in model.capsule_layers
    ]

    out_vect = [
        layer.last_caps_output.numpy()
        for layer in model.capsule_layers
    ]

    pred = model.get_scores(
        capsule_output
    ).numpy()

    # --------------------------------------------------------
    # Estrazione regole del batch
    # --------------------------------------------------------

    batch_rules = extract_rules_boundary(
        X_batch.numpy(),
        coupl_coeff,
        pred_vect,
        out_vect,
        pred,
    )

    rule_set.append(
        batch_rules
    )

    print(
        f"Batch {batch_number}/{num_training_batches} "
        f"-> {len(batch_rules)} gruppi di regole"
    )


# ============================================================
# SALVATAGGIO RULE SET COMUNE
# ============================================================

rule_set_path = (
    experiments_dir / "ablation_rule_set_initial.joblib"
)

joblib.dump(
    rule_set,
    rule_set_path,
)

selected_features_path = (
    experiments_dir / "ablation_selected_features.joblib"
)

joblib.dump(
    selected_features,
    selected_features_path,
)

print(
    "\nRule set comune salvato in:",
    rule_set_path,
)

print(
    "Feature selezionate salvate in:",
    selected_features_path,
)

print(
    "Numero gruppi estratti:",
    len(rule_set),
)


# ============================================================
# PREPARAZIONE LABEL ONE-HOT PER LA VALIDAZIONE
# ============================================================

y_val_one_hot = tf.one_hot(
    y_val,
    depth=N_CLASSES,
).numpy()


# ============================================================
# ABLATION A: NO SHAP
# ============================================================
#
# STESSO:
#   - FFCN
#   - Rule Set
#   - validation set
#   - threshold precision
#   - threshold support
#
# DIFFERENZA:
#   shap_allowed_features = None
# ============================================================

print("\n============================================================")
print("=== ABLATION A: FFCN + ATTACK RULES SENZA SHAP ===")
print("============================================================")

validated_rules_no_shap = validate_rules(
    rule_set,
    X_val.numpy(),
    y_val_one_hot,
    min_precision=0.85,
    min_support=3,
    shap_allowed_features=None,
)

print(
    "\nRegole validate NO SHAP:",
    len(validated_rules_no_shap),
)

validated_rules_no_shap_path = (
    experiments_dir / "ablation_validated_rules_no_shap.joblib"
)

joblib.dump(
    validated_rules_no_shap,
    validated_rules_no_shap_path,
)

print(
    "Regole NO SHAP salvate in:",
    validated_rules_no_shap_path,
)


# ============================================================
# TEST A: HYBRID NO SHAP
# ============================================================

y_test_one_hot = tf.one_hot(
    y_test,
    depth=N_CLASSES,
).numpy()

print("\n============================================================")
print("=== TEST HYBRID NO SHAP ===")
print("============================================================")

evaluate_rules_boundary(
    validated_rules_no_shap,
    X_test.numpy(),
    y_test_one_hot,
    ffcn_predictions=test_predictions,
)

print(
    "\nEvaluation Hybrid NO SHAP completata."
)


# ============================================================
# ABLATION B: SHAP
# ============================================================
#
# STESSO:
#   - FFCN
#   - Rule Set
#   - validation set
#   - threshold precision
#   - threshold support
#
# DIFFERENZA:
#   viene applicato il filtro SHAP sulle feature delle ATTACK
#   rules.
# ============================================================

print("\n============================================================")
print("=== ABLATION B: FFCN + ATTACK RULES CON SHAP ===")
print("============================================================")

validated_rules_shap = validate_rules(
    rule_set,
    X_val.numpy(),
    y_val_one_hot,
    min_precision=0.85,
    min_support=3,
    shap_allowed_features=shap_allowed_features,
)

print(
    "\nRegole validate SHAP:",
    len(validated_rules_shap),
)

validated_rules_shap_path = (
    experiments_dir / "ablation_validated_rules_shap.joblib"
)

joblib.dump(
    validated_rules_shap,
    validated_rules_shap_path,
)

print(
    "Regole SHAP salvate in:",
    validated_rules_shap_path,
)


# ============================================================
# TEST B: HYBRID SHAP
# ============================================================

print("\n============================================================")
print("=== TEST HYBRID SHAP ===")
print("============================================================")

evaluate_rules_boundary(
    validated_rules_shap,
    X_test.numpy(),
    y_test_one_hot,
    ffcn_predictions=test_predictions,
)

print(
    "\nEvaluation Hybrid SHAP completata."
)


# ============================================================
# RIEPILOGO ABLATION
# ============================================================

print("\n============================================================")
print("=== RIEPILOGO ABLATION STUDY ===")
print("============================================================")

print("\nFFCΝ COMUNE A ENTRAMBI:")
print(f"Precision: {ffcn_precision:.4f}")
print(f"Recall:    {ffcn_recall:.4f}")
print(f"F1:        {ffcn_f1:.4f}")

print("\nRULE SET CANDIDATO COMUNE:")
print(
    "Gruppi di regole estratti:",
    len(rule_set),
)

print("\nNO SHAP:")
print(
    "Regole validate:",
    len(validated_rules_no_shap),
)

print("\nSHAP:")
print(
    "Regole validate:",
    len(validated_rules_shap),
)

print("\n============================================================")
print("IMPORTANTE: NO SHAP e SHAP hanno usato LO STESSO:")
print("- FFCN addestrato")
print("- best checkpoint")
print("- train/validation/test split")
print("- Rule Set candidato")
print("- soglia min_precision")
print("- soglia min_support")
print("- test set")
print("============================================================")
