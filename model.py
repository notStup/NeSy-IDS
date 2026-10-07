import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
from preprocessing import create_split, load_split, save_split
from squash import squash, safe_norm
from rule_extraction import extract_rules_boundary
from rule_validation import validate_rules
from rule_evaluation import evaluate_rules_boundary
from shap_guidance import compute_attack_shap_importance, get_top_shap_features, show_shap, logic_shap, print_shap_rules
import numpy as np
import joblib



# ESPERIMENTO

# True:
#     carica uno split già salvato
#
# False:
#     crea un nuovo split e lo salva

LOAD_SAVED_SPLIT = True


# Nome univoco dell'esperimento.
#
# Questo nome viene usato anche per associare:
# - split
# - modello
# - SHAP
# - rules

SPLIT_NAME = (
    "E1_train2000_val5000_test5000_seed42"
)


# ------------------------------------------------------------
# Parametri utilizzati SOLO quando si crea un nuovo split
# ------------------------------------------------------------

TRAIN_SAMPLES = 2000

VAL_SAMPLES = 5000

TEST_SAMPLES = 5000

RANDOM_STATE = 42


# ============================================================
# MODELLO
# ============================================================

BATCH_SIZE = 100

N_CLASSES = 2

# 0 = BENIGN
# 1 = ATTACK

SIZE_LAYERS = [
    25,
    20,
    15,
    8,
    N_CLASSES
]

CAPS_DIMS = [
    30,
    25,
    20,
    15,
    10
]

ROUTING_ITERATIONS = 3

LEARNING_RATE = 0.001

EPOCHS = 10

RECONSTRUCTION_COEFFICIENT = 5.0e-15


# ============================================================
# MODELLO SALVATO
# ============================================================

# True:
#     usa il modello già salvato se disponibile
#
# False:
#     forza un nuovo training

LOAD_SAVED_MODEL = True


# ============================================================
# SHAP
# ============================================================

USE_SHAP_RULE_FILTER = True

# Se True, usa SHAP già salvato.
# Se False, ricalcola SHAP.

LOAD_SAVED_SHAP = False

SHAP_BACKGROUND_SAMPLES = 100

SHAP_EXPLAIN_SAMPLES = 2000

SHAP_NSAMPLES = 200

# Numero di feature SHAP utilizzate
# per filtrare le ATTACK rules.

SHAP_TOP_K = 35


# ============================================================
# REGOLE
# ============================================================

# Se True, usa le rules già salvate.
# Se False, forza una nuova estrazione.

LOAD_SAVED_RULES = False


MIN_RULE_PRECISION = 0.80

MIN_RULE_SUPPORT = 2


# ============================================================
# CARTELLE ESPERIMENTO
# ============================================================

experiment_dir = (
    Path("Experiments")
    / SPLIT_NAME
)

model_dir = (
    experiment_dir
    / "model"
)

shap_dir = (
    experiment_dir
    / "shap"
)

rules_dir = (
    experiment_dir
    / "rules"
)


experiment_dir.mkdir(
    parents=True,
    exist_ok=True
)

model_dir.mkdir(
    parents=True,
    exist_ok=True
)

shap_dir.mkdir(
    parents=True,
    exist_ok=True
)

rules_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATASET / SPLIT
# ============================================================

print("\n=== DATASET / SPLIT ===")


if LOAD_SAVED_SPLIT:

    # --------------------------------------------------------
    # CARICAMENTO
    # --------------------------------------------------------

    split = load_split(
        SPLIT_NAME
    )

else:

    # --------------------------------------------------------
    # CREAZIONE
    # --------------------------------------------------------

    split = create_split(
        train_samples=TRAIN_SAMPLES,
        val_samples=VAL_SAMPLES,
        test_samples=TEST_SAMPLES,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # SALVATAGGIO
    # --------------------------------------------------------

    save_split(
        split,
        SPLIT_NAME
    )


# ============================================================
# ESTRAZIONE DATASET
# ============================================================

X_train = split["X_train"]

y_train = split["y_train"]

X_val = split["X_val"]

y_val = split["y_val"]

X_test = split["X_test"]

y_test = split["y_test"]


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = split["feature_names"]

# Tutte le feature sono utilizzate dalla FFCN.

INPUT_SIZE = len(
    feature_names
)


print(
    "\n=== EXPERIMENT SPLIT ==="
)

print(
    "X_train:",
    X_train.shape
)

print(
    "X_val:",
    X_val.shape
)

print(
    "X_test:",
    X_test.shape
)

print(
    "Numero feature:",
    INPUT_SIZE
)


# Salvataggio dei nomi delle feature
# nello spazio dell'esperimento.

feature_names_path = (
    experiment_dir
    / "feature_names.joblib"
)

joblib.dump(
    feature_names,
    feature_names_path
)


# ============================================================
# CONVERSIONE TENSORFLOW
# ============================================================

X_train = tf.convert_to_tensor(
    X_train.to_numpy(),
    dtype=tf.float32
)

X_val = tf.convert_to_tensor(
    X_val.to_numpy(),
    dtype=tf.float32
)

X_test = tf.convert_to_tensor(
    X_test.to_numpy(),
    dtype=tf.float32
)


y_train = tf.convert_to_tensor(
    y_train.to_numpy().reshape(-1),
    dtype=tf.int32
)

y_val = tf.convert_to_tensor(
    y_val.to_numpy().reshape(-1),
    dtype=tf.int32
)

y_test = tf.convert_to_tensor(
    y_test.to_numpy().reshape(-1),
    dtype=tf.int32
)


# ============================================================
# INPUT CAPSULE
# ============================================================

# (batch, n_features)
#          ↓
# (batch, n_features, 1)

X_train = tf.expand_dims(
    X_train,
    axis=-1
)

X_val = tf.expand_dims(
    X_val,
    axis=-1
)

X_test = tf.expand_dims(
    X_test,
    axis=-1
)


print(
    "\nInput capsule:"
)

print(
    "X_train:",
    X_train.shape
)

print(
    "X_val:",
    X_val.shape
)

print(
    "X_test:",
    X_test.shape
)


# ============================================================
# CAPSULE LAYER
# ============================================================

class CapsuleLayer(layers.Layer):

    def __init__(
        self,
        num_capsules,
        capsule_dim,
        routing_iterations=3
    ):

        super().__init__()

        self.num_capsules = (
            num_capsules
        )

        self.capsule_dim = (
            capsule_dim
        )

        self.routing_iterations = (
            routing_iterations
        )

        # Informazioni utilizzate da CapsRule.

        self.last_coupling_coefficients = (
            None
        )

        self.last_caps_predicted = (
            None
        )

        self.last_caps_output = (
            None
        )


    def build(
        self,
        input_shape
    ):

        self.input_capsules = int(
            input_shape[1]
        )

        self.input_dim = int(
            input_shape[2]
        )

        self.W = self.add_weight(

            name="weights",

            shape=(

                1,

                self.input_capsules,

                self.num_capsules,

                self.capsule_dim,

                self.input_dim

            ),

            initializer=(
                tf.keras.initializers
                .RandomNormal(
                    mean=0.0,
                    stddev=1.0
                )
            ),

            trainable=True
        )


    def call(
        self,
        inputs,
        training=False
    ):

        # ====================================================
        # PREDICTION VECTORS
        # ====================================================

        inputs64 = tf.cast(
            inputs,
            tf.float64
        )

        W64 = tf.cast(

            tf.squeeze(
                self.W,
                axis=0
            ),

            tf.float64
        )

        caps_predicted = tf.einsum(

            "bid,ijod->bijo",

            inputs64,

            W64

        )


        # ====================================================
        # INITIAL ROUTING WEIGHTS
        # ====================================================

        batch_size = (
            tf.shape(inputs64)[0]
        )

        raw_weights = tf.zeros(

            (

                batch_size,

                self.input_capsules,

                self.num_capsules

            ),

            dtype=tf.float64
        )


        coupling_coefficients = None

        caps_output = None


        # ====================================================
        # DYNAMIC ROUTING
        # ====================================================

        for _ in range(
            self.routing_iterations
        ):

            coupling_coefficients = (
                tf.nn.softmax(
                    raw_weights,
                    axis=-1
                )
            )

            weighted_predictions = (

                coupling_coefficients[
                    ..., tf.newaxis
                ]

                * caps_predicted

            )

            weighted_sum = (
                tf.reduce_sum(
                    weighted_predictions,
                    axis=1
                )
            )

            caps_output = squash(
                weighted_sum,
                axis=-1
            )

            agreement = (
                tf.reduce_sum(

                    caps_predicted
                    * caps_output[
                        :,
                        tf.newaxis,
                        :,
                        :
                    ],

                    axis=-1
                )
            )

            raw_weights = (
                raw_weights
                + agreement
            )


        # ====================================================
        # FINAL COUPLING COEFFICIENTS
        # ====================================================

        coupling_coefficients = (
            tf.nn.softmax(
                raw_weights,
                axis=-1
            )
        )


        # ====================================================
        # SALVATAGGIO CAPSRule
        # ====================================================

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
            inputs.dtype
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
                routing_iterations=ROUTING_ITERATIONS
            ),

            CapsuleLayer(
                num_capsules=SIZE_LAYERS[1],
                capsule_dim=CAPS_DIMS[1],
                routing_iterations=ROUTING_ITERATIONS
            ),

            CapsuleLayer(
                num_capsules=SIZE_LAYERS[2],
                capsule_dim=CAPS_DIMS[2],
                routing_iterations=ROUTING_ITERATIONS
            ),

            CapsuleLayer(
                num_capsules=SIZE_LAYERS[3],
                capsule_dim=CAPS_DIMS[3],
                routing_iterations=ROUTING_ITERATIONS
            ),

            CapsuleLayer(
                num_capsules=SIZE_LAYERS[4],
                capsule_dim=CAPS_DIMS[4],
                routing_iterations=ROUTING_ITERATIONS
            )

        ]


        # ====================================================
        # DECODER
        # ====================================================

        self.decoder = (
            tf.keras.Sequential([

                layers.Dense(
                    5,
                    activation="relu"
                ),

                layers.Dense(
                    15,
                    activation="relu"
                ),

                layers.Dense(
                    20,
                    activation="relu"
                ),

                layers.Dense(
                    40,
                    activation="relu"
                ),

                layers.Dense(
                    INPUT_SIZE,
                    activation="sigmoid"
                )

            ])
        )


        # ====================================================
        # METRICHE
        # ====================================================

        self.loss_tracker = (
            tf.keras.metrics.Mean(
                name="loss"
            )
        )

        self.margin_loss_tracker = (
            tf.keras.metrics.Mean(
                name="margin_loss"
            )
        )

        self.reconstruction_loss_tracker = (
            tf.keras.metrics.Mean(
                name="reconstruction_loss"
            )
        )

        self.accuracy_tracker = (
            tf.keras.metrics.Mean(
                name="accuracy"
            )
        )


    @property
    def metrics(self):

        return [

            self.loss_tracker,

            self.margin_loss_tracker,

            self.reconstruction_loss_tracker,

            self.accuracy_tracker

        ]


    def call(
        self,
        inputs,
        training=False
    ):

        x = inputs

        for capsule_layer in (
            self.capsule_layers
        ):

            x = capsule_layer(
                x,
                training=training
            )

        return x


    def get_scores(
        self,
        capsule_output
    ):

        return safe_norm(
            capsule_output,
            axis=-1
        )


    def get_predictions(
        self,
        capsule_output
    ):

        scores = self.get_scores(
            capsule_output
        )

        return tf.argmax(

            scores,

            axis=-1,

            output_type=tf.int32

        )


    def reconstruct(
        self,
        capsule_output,
        y_true
    ):

        y_one_hot = tf.one_hot(

            y_true,

            depth=N_CLASSES,

            dtype=capsule_output.dtype

        )

        mask = (
            y_one_hot[..., tf.newaxis]
        )

        masked_capsules = (
            capsule_output
            * mask
        )

        decoder_input = tf.reshape(

            masked_capsules,

            (
                -1,
                N_CLASSES * CAPS_DIMS[-1]
            )

        )

        return self.decoder(
            decoder_input
        )


    def compute_losses(
        self,
        inputs,
        y_true,
        capsule_output
    ):

        margin = margin_loss(

            y_true,

            capsule_output

        )

        reconstruction_output = (
            self.reconstruct(
                capsule_output,
                y_true
            )
        )

        input_flat = tf.reshape(

            inputs,

            (
                tf.shape(inputs)[0],
                INPUT_SIZE
            )

        )

        reconstruction_loss = (
            tf.reduce_mean(

                tf.square(

                    input_flat
                    - reconstruction_output

                )

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
            reconstruction_loss

        )


    def train_step(
        self,
        data
    ):

        inputs, y_true = data

        y_true = tf.cast(

            tf.reshape(
                y_true,
                [-1]
            ),

            tf.int32

        )

        with tf.GradientTape() as tape:

            capsule_output = self(

                inputs,

                training=True

            )

            (
                total_loss,
                margin,
                reconstruction
            ) = self.compute_losses(

                inputs,

                y_true,

                capsule_output

            )


        gradients = tape.gradient(

            total_loss,

            self.trainable_variables

        )


        self.optimizer.apply_gradients(

            zip(
                gradients,
                self.trainable_variables
            )

        )


        predictions = (
            self.get_predictions(
                capsule_output
            )
        )


        accuracy = tf.reduce_mean(

            tf.cast(

                tf.equal(
                    predictions,
                    y_true
                ),

                tf.float32

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
                self.accuracy_tracker.result()

        }


    def test_step(
        self,
        data
    ):

        inputs, y_true = data

        y_true = tf.cast(

            tf.reshape(
                y_true,
                [-1]
            ),

            tf.int32

        )


        capsule_output = self(

            inputs,

            training=False

        )


        (
            total_loss,
            margin,
            reconstruction
        ) = self.compute_losses(

            inputs,

            y_true,

            capsule_output

        )


        predictions = (
            self.get_predictions(
                capsule_output
            )
        )


        accuracy = tf.reduce_mean(

            tf.cast(

                tf.equal(
                    predictions,
                    y_true
                ),

                tf.float32

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
                self.accuracy_tracker.result()

        }


# ============================================================
# MARGIN LOSS
# ============================================================

def margin_loss(
    y_true,
    y_pred
):

    capsule_lengths = safe_norm(

        y_pred,

        axis=-1

    )


    y_true = tf.one_hot(

        tf.cast(
            y_true,
            tf.int32
        ),

        depth=N_CLASSES,

        dtype=capsule_lengths.dtype

    )


    m_plus = 0.9

    m_minus = 0.1

    lambda_ = 0.5


    positive_loss = (

        y_true

        * tf.square(

            tf.maximum(

                0.0,

                m_plus
                - capsule_lengths

            )

        )

    )


    negative_loss = (

        lambda_

        * (1.0 - y_true)

        * tf.square(

            tf.maximum(

                0.0,

                capsule_lengths
                - m_minus

            )

        )

    )


    return tf.reduce_mean(

        tf.reduce_sum(

            positive_loss
            + negative_loss,

            axis=-1

        )

    )


# ============================================================
# CREAZIONE MODELLO
# ============================================================

model = FFCN()


# ============================================================
# BUILD ESPLICITO
# ============================================================

dummy_output = model(

    X_train[:BATCH_SIZE],

    training=False

)

print(
    "\nOutput FFCN:",
    dummy_output.shape
)


# ============================================================
# OPTIMIZER
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=LEARNING_RATE,

        global_clipnorm=1.0

    )

)


# ============================================================
# CHECKPOINT
# ============================================================

checkpoint_path = (

    model_dir
    / "best_ffcn.weights.h5"

)


checkpoint = (
    tf.keras.callbacks.ModelCheckpoint(

        filepath=str(
            checkpoint_path
        ),

        monitor="val_loss",

        mode="min",

        save_best_only=True,

        save_weights_only=True,

        verbose=1

    )
)


early_stopping = (
    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        mode="min",

        patience=10,

        restore_best_weights=True,

        verbose=1

    )
)


# ============================================================
# TRAINING / CARICAMENTO MODELLO
# ============================================================

if (
    LOAD_SAVED_MODEL
    and checkpoint_path.exists()
):

    print(
        "\n=== MODELLO ESISTENTE ==="
    )

    print(
        "Caricamento pesi da:",
        checkpoint_path
    )

    model.load_weights(
        str(checkpoint_path)
    )

    print(
        "Modello caricato."
    )


else:

    print(
        "\n=== TRAINING FFCN ==="
    )

    history = model.fit(

        X_train,

        y_train,

        validation_data=(

            X_val,

            y_val

        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=[

            checkpoint,

            early_stopping

        ]

    )


    model.load_weights(
        str(checkpoint_path)
    )

    print(
        "\nMigliori pesi caricati da:",
        checkpoint_path
    )


# ============================================================
# DIAGNOSTICA FFCN
# ============================================================

print(
    "\n=== DIAGNOSTICA FFCN ==="
)


val_output = model(

    X_val,

    training=False

)


val_predictions = (

    model
    .get_predictions(
        val_output
    )
    .numpy()

)


test_output = model(

    X_test,

    training=False

)


y_pred = (

    model
    .get_predictions(
        test_output
    )
    .numpy()

)


y_test_np = y_test.numpy()


# ============================================================
# FFCN METRICS
# ============================================================

ffcn_precision = precision_score(

    y_test_np,

    y_pred,

    pos_label=1,

    zero_division=0

)


ffcn_recall = recall_score(

    y_test_np,

    y_pred,

    pos_label=1,

    zero_division=0

)


ffcn_f1 = f1_score(

    y_test_np,

    y_pred,

    pos_label=1,

    zero_division=0

)


print(
    "\n=== FFCN ATTACK METRICS ==="
)


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
# SHAP
# ============================================================

shap_allowed_features = None

shap_values = None

X_explain = None


if USE_SHAP_RULE_FILTER:

    shap_results_path = (

        shap_dir
        / "shap_results.joblib"

    )

    shap_plot_data_path = (

        shap_dir
        / "shap_plot_data.joblib"

    )


    if (
        LOAD_SAVED_SHAP
        and shap_results_path.exists()
    ):

        print(
            "\n=== CARICAMENTO SHAP ==="
        )


        shap_results = joblib.load(

            shap_results_path

        )


        shap_importance = (
            shap_results["importance"]
        )

        shap_ranking = (
            shap_results["ranking"]
        )

        shap_allowed_features = (
            shap_results["allowed_features"]
        )


        if shap_plot_data_path.exists():

            shap_plot_data = joblib.load(

                shap_plot_data_path

            )

            shap_values = (
                shap_plot_data["shap_values"]
            )

            X_explain = (
                shap_plot_data["X_explain"]
            )


    else:

        print(
            "\n=== SHAP FEATURE IMPORTANCE ==="
        )


        (
            shap_importance,
            shap_ranking,
            shap_values,
            X_explain
        ) = compute_attack_shap_importance(

            model,

            X_train,

            y_train,

            background_samples=(
                SHAP_BACKGROUND_SAMPLES
            ),

            explain_samples=(
                SHAP_EXPLAIN_SAMPLES
            ),

            nsamples=(
                SHAP_NSAMPLES
            ),

            random_state=RANDOM_STATE

        )


        shap_allowed_features = (
            get_top_shap_features(

                shap_importance,

                SHAP_TOP_K

            )
        )


        shap_results = {

            "importance":
                shap_importance,

            "ranking":
                shap_ranking,

            "top_k":
                SHAP_TOP_K,

            "allowed_features":
                sorted(
                    shap_allowed_features
                )

        }


        joblib.dump(

            shap_results,

            shap_results_path

        )


        joblib.dump(

            {

                "shap_values":
                    shap_values,

                "X_explain":
                    X_explain

            },

            shap_plot_data_path

        )


    # ========================================================
    # STAMPA SHAP
    # ========================================================

    print(
        "\nTop feature SHAP:"
    )


    for rank, feature_index in enumerate(

        shap_ranking[
            :SHAP_TOP_K
        ],

        start=1

    ):

        feature_name = (

            feature_names[
                feature_index
            ]

            if feature_index
            < len(feature_names)

            else
            f"Feature {feature_index}"

        )


        print(

            f"{rank}. "
            f"Feature {feature_index} "
            f"({feature_name}) "
            f"importance="
            f"{shap_importance[feature_index]:.6f}"

        )


    print(

        "\nFeature utilizzabili "
        "dalle ATTACK rules:",

        sorted(
            shap_allowed_features
        )

    )


    if (
        shap_values is not None
        and X_explain is not None
    ):

        show_shap(

            shap_values,

            X_explain,

            feature_names,
            shap_dir

        )


# ============================================================
# RULE EXTRACTION
# ============================================================

rule_set_path = (

    rules_dir
    / "rule_set_initial.joblib"

)

validated_rules_path = (

    rules_dir
    / "validated_rules.joblib"

)


if (
    LOAD_SAVED_RULES
    and rule_set_path.exists()
    and validated_rules_path.exists()
):

    # ========================================================
    # CARICAMENTO RULES
    # ========================================================

    print(
        "\n=== REGOLE GIÀ DISPONIBILI ==="
    )


    rule_set = joblib.load(

        rule_set_path

    )


    validated_rules = joblib.load(

        validated_rules_path

    )


    print(

        "Rule set caricato:",

        len(rule_set),

        "regole"

    )


    print(

        "Regole validate:",

        len(validated_rules)

    )


else:

    # ========================================================
    # ESTRAZIONE CAPSRule
    # ========================================================

    print(
        "\n=== ESTRAZIONE CAPSRule ==="
    )


    rule_arr_class = None


    num_extraction_batches = (

        (
            len(X_train)
            + BATCH_SIZE
            - 1
        )
        // BATCH_SIZE

    )


    for batch_number, start in enumerate(

        range(
            0,
            len(X_train),
            BATCH_SIZE
        ),

        start=1

    ):

        end = min(

            start
            + BATCH_SIZE,

            len(X_train)

        )


        X_batch = (
            X_train[start:end]
        )


        capsule_output = model(

            X_batch,

            training=False

        )


        coupl_coeff = [

            layer
            .last_coupling_coefficients
            .numpy()

            for layer
            in model.capsule_layers

        ]


        pred_vect = [

            layer
            .last_caps_predicted
            .numpy()

            for layer
            in model.capsule_layers

        ]


        out_vect = [

            layer
            .last_caps_output
            .numpy()

            for layer
            in model.capsule_layers

        ]


        pred = (

            model
            .get_scores(
                capsule_output
            )
            .numpy()

        )


        rule_arr_class = (
            extract_rules_boundary(

                X_batch.numpy(),

                coupl_coeff,

                pred_vect,

                out_vect,

                pred,

                feature_names,

                rule_arr_class

            )
        )


        print(

            f"Batch di estrazione "
            f"{batch_number}/"
            f"{num_extraction_batches} "
            f"completato"

        )


    # ========================================================
    # GENERAZIONE RULE SET
    # ========================================================

    rule_set = []


    for class_index in range(
        N_CLASSES
    ):

        for rule_dict in (
            rule_arr_class[
                class_index
            ]
        ):

            class_rule = [
                "("
            ]


            for feature_index in (
                rule_dict
            ):

                class_rule.append(

                    str(

                        min(

                            rule_dict[
                                feature_index
                            ]

                        )

                    )

                )


                class_rule.append(
                    "<="
                )


                class_rule.append(

                    feature_names[
                        feature_index
                    ]

                )


                class_rule.append(
                    "Feature:"
                )


                class_rule.append(

                    int(
                        feature_index
                    )

                )


                class_rule.append(
                    "<="
                )


                class_rule.append(

                    str(

                        max(

                            rule_dict[
                                feature_index
                            ]

                        )

                    )

                )


                class_rule.append(
                    "and"
                )


            if (
                class_rule[-1]
                == "and"
            ):

                class_rule.pop()


            class_rule.append(
                ")"
            )


            class_rule.append(
                class_index
            )


            rule_set.append(
                class_rule
            )


    # ========================================================
    # SALVATAGGIO RULE SET
    # ========================================================

    joblib.dump(

        rule_set,

        rule_set_path

    )


    print(

        "\nRule set salvato in:",

        rule_set_path

    )


    print(

        "Numero gruppi estratti:",

        len(rule_set)

    )

  
    # VALIDAZIONE REGOLE

    print(
        "\n=== VALIDAZIONE REGOLE ==="
    )


    y_val_one_hot = tf.one_hot(

        y_val,

        depth=N_CLASSES

    ).numpy()




    validated_rules, rule_metadata = validate_rules(

        rule_set,

        X_val.numpy(),

        y_val_one_hot,

        min_precision=MIN_RULE_PRECISION,

        min_support=MIN_RULE_SUPPORT,

        shap_importance= shap_importance,
        return_metadata= True
        

    )
    
    
      # ========================================================
    # ANALISI SHAP DELLE REGOLE
    # ========================================================
    
    print(
        "\n=== SHAP ANALYSIS DELLE REGOLE ==="
    )

    rule_shap_results = logic_shap(

        X_explain,

        shap_values,

        validated_rules,

    )
    
    print_shap_rules(validated_rules, rule_shap_results, feature_names)

    



    print(

        "\nRegole validate:",

        len(validated_rules)

    )


    # SALVATAGGIO REGOLE VALIDATE

    joblib.dump(

        validated_rules,

        validated_rules_path

    )


    print(

        "Regole validate salvate in:",

        validated_rules_path

    )


# EVALUATION HYBRID SUL TEST

print(
    "\n=== TEST DELLE REGOLE ==="
)


y_test_one_hot = tf.one_hot(

    y_test,

    depth=N_CLASSES

).numpy()


evaluate_rules_boundary(

    validated_rules,

    X_test.numpy(),

    y_test_one_hot,

    ffcn_predictions=y_pred

)


# STAMPA REGOLE

for i, rule in enumerate(

    validated_rules,

    start=1

):

    print(
        i,
        rule
    )


print(

    "\nRegole validate:",

    len(validated_rules)

)


print(
    "\nEvaluation delle regole completata."
)