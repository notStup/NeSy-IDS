import pandas as pd
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import train_test_split


# CONFIGURAZIONE

data_dir = Path("data/CICIDS-2017")


# CARICAMENTO DATASET

def load_dataset(data_dir):

    csv_files = sorted(
        data_dir.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"Nessun CSV trovato nella cartella: {data_dir}"
        )

    dataframes = []

    for csv_file in csv_files:

        print(
            f"Caricamento: {csv_file.name}"
        )

        df = pd.read_csv(
            csv_file,
            low_memory=False
        )

        df.columns = (
            df.columns
            .str.strip()
        )

        dataframes.append(df)

    dataset = pd.concat(
        dataframes,
        ignore_index=True
    )

    print(
        f"Dataset totale: "
        f"{dataset.shape[0]:,} righe"
    )

    print(
        f"Numero colonne: "
        f"{dataset.shape[1]}"
    )

    return dataset


# PULIZIA DATASET

def clean_dataset(df):

    df = df.copy()

    columns_to_remove = [
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp",
        "Protocol"
    ]

    df = df.drop(
        columns=columns_to_remove,
        errors="ignore"
    )

    feature_columns = [
        col
        for col in df.columns
        if col != "Label"
    ]

    for column in feature_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df[feature_columns] = (
        df[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    df[feature_columns] = (
        df[feature_columns]
        .fillna(0)
    )

    return df


# PREPARAZIONE LABEL

def prepare_label(df):

    df = df.copy()

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )

    df["BinaryLabel"] = (
        df["Label"].str.upper()
        != "BENIGN"
    ).astype(int)

    return df


# INFORMAZIONI DATASET

def show_dataset_info(df):

    print("\n=== DATASET INFO ===")

    print("\nDistribuzione Label:")

    print(
        df["Label"].value_counts()
    )

    print("\nDistribuzione BinaryLabel:")

    print(
        df["BinaryLabel"].value_counts()
    )

    feature_names = [
        col
        for col in df.columns
        if col not in [
            "Label",
            "BinaryLabel"
        ]
    ]

    print(
        "\nNumero feature:",
        len(feature_names)
    )




# CONFIGURAZIONE SPLIT

SPLIT_DIR = data_dir / "splits"
SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# CREAZIONE SPLIT

def create_split(
    train_samples,
    val_samples,
    test_samples,
    random_state=42
):
    """
    Crea uno split sperimentale completo:

    - train
    - validation
    - test

    Tutte le feature vengono mantenute.
    """

    # Caricamento dataset

    df = load_dataset(
        data_dir
    )

    # Pulizia

    df = clean_dataset(
        df
    )

    # Label binaria

    df = prepare_label(
        df
    )

    # Informazioni

    show_dataset_info(
        df
    )

    # Separazione X / y

    X = df.drop(
        columns=[
            "Label",
            "BinaryLabel"
        ]
    )

    y = df["BinaryLabel"]

    # Train / Test base

    X_train_full, X_test_full, y_train_full, y_test_full = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=random_state,
            stratify=y
        )
    )

    # Train / Validation

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.10,
            random_state=random_state,
            stratify=y_train_full
        )
    )

    # Riduzione TRAIN

    if train_samples > len(X_train):

        raise ValueError(
            f"train_samples={train_samples} "
            f"ma il training disponibile è "
            f"{len(X_train)}"
        )

    X_train, _, y_train, _ = (
        train_test_split(
            X_train,
            y_train,
            train_size=train_samples,
            random_state=random_state,
            stratify=y_train
        )
    )

    # Riduzione VALIDATION

    if val_samples > len(X_val):

        raise ValueError(
            f"val_samples={val_samples} "
            f"ma la validation disponibile è "
            f"{len(X_val)}"
        )

    X_val, _, y_val, _ = (
        train_test_split(
            X_val,
            y_val,
            train_size=val_samples,
            random_state=random_state,
            stratify=y_val
        )
    )

    # Riduzione TEST

    if test_samples > len(X_test_full):

        raise ValueError(
            f"test_samples={test_samples} "
            f"ma il test disponibile è "
            f"{len(X_test_full)}"
        )

    X_test, _, y_test, _ = (
        train_test_split(
            X_test_full,
            y_test_full,
            train_size=test_samples,
            random_state=random_state,
            stratify=y_test_full
        )
    )

    # Feature names
    

    feature_names = list(
        X_train.columns
    )

    # Risultato

    split = {

        "X_train": X_train,
        "y_train": y_train,

        "X_val": X_val,
        "y_val": y_val,

        "X_test": X_test,
        "y_test": y_test,

        "feature_names": feature_names,

        "config": {
            "train_samples": train_samples,
            "val_samples": val_samples,
            "test_samples": test_samples,
            "random_state": random_state,
            "test_fraction": 0.20,
            "validation_fraction": 0.10
        }
    }

    # Diagnostica

    print("\n=== SPLIT CREATO ===")

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
        len(feature_names)
    )

    return split


# SALVATAGGIO SPLIT

def save_split(
    split,
    name
):
    """
    Salva uno split sperimentale.
    """

    split_path = (
        SPLIT_DIR /
        f"{name}.joblib"
    )

    joblib.dump(
        split,
        split_path,
        compress=3
    )

    print(
        f"\nSplit salvato in:"
        f" {split_path}"
    )

    return split_path


# CARICAMENTO SPLIT

def load_split(
    name
):
    """
    Carica uno split sperimentale esistente.
    """

    split_path = (
        SPLIT_DIR /
        f"{name}.joblib"
    )

    if not split_path.exists():

        raise FileNotFoundError(
            f"Split non trovato:"
            f" {split_path}"
        )

    split = joblib.load(
        split_path
    )

    print(
        f"\nSplit caricato da:"
        f" {split_path}"
    )

    print(
        "\nConfigurazione:"
    )

    print(
        split["config"]
    )

    print(
        "\nX_train:",
        split["X_train"].shape
    )

    print(
        "X_val:",
        split["X_val"].shape
    )

    print(
        "X_test:",
        split["X_test"].shape
    )

    return split