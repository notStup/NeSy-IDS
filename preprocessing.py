
import pandas as pd 
from pathlib import Path
import numpy as np
import joblib
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

data_dir= Path("data/CICIDS-2017")
cache_file = data_dir / "processed_30_features.joblib"

def load_dataset(data_dir):

    csv_files= list(data_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(

            "Nessun CSV trovato nella cartella: {data_dir}"

        ) 


    dataframes= []



    for csv_file in csv_files:
        print(f"Caricamento: {csv_file.name}")
        
        df= pd.read_csv(csv_file, low_memory=False)
        df.columns = df.columns.str.strip()
        dataframes.append(df)
        
    dataset= pd.concat(dataframes, ignore_index= True)
    
    print(f"Dataset totale: {dataset.shape[0]:,} righe")
    print(f"Numero feature: {dataset.shape[1]} ")
    
    return dataset




def clean_dataset(df):
    
    df= df.copy()
    
    #Eliminiamo queste colonne perché perché gli indirizzi IP dipendono molto dalla configurazione della rete e possono portare a overfitting. A6.pdf
    
    columns_to_remove = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
]
    


    # Alcune versioni dei CSV CICIDS-2017 non includono queste colonne.
    df = df.drop(columns=columns_to_remove, errors="ignore")
        
    # Converti le feature prima di cercare NaN/inf: nei CSV alcuni numeri
    # possono essere letti come stringhe e sfuggire al controllo numerico.
    feature_columns = [col for col in df.columns if col != "Label"]
    for column in feature_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    df[feature_columns] = df[feature_columns].fillna(0)

    return df


#Preparo la label rendendola binaria, tutto ciò che non è benigno è maligno -> 1
def prepare_label(df):
    
    df["Label"] = df["Label"].astype(str).str.strip()
    
    df["BinaryLabel"] = (
    df["Label"].str.upper() != "BENIGN").astype(int)

    return df




def split_dataset(df):

    X = df.drop(columns=["Label", "BinaryLabel"])

    y = df["BinaryLabel"]

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    return X_train, X_test, y_train, y_test



def select_top_features(X_train, X_test, y_train, n_features=30, sample_size=100_000):

    # La MI è costosa sui milioni di record: stimiamo i punteggi su un
    # campione stratificato, mantenendo tutti i record per il modello.
    if len(X_train) > sample_size:
        X_sample, _, y_sample, _ = train_test_split(
            X_train,
            y_train,
            train_size=sample_size,
            random_state=42,
            stratify=y_train,
        )
    else:
        X_sample = X_train
        y_sample = y_train

    mi_scores = mutual_info_classif(

        X_sample,

        y_sample,

        random_state=42
    )

    mi_scores = pd.Series(

        mi_scores,

        index=X_train.columns

    ).sort_values(ascending=False)

    selected_features = mi_scores.head(n_features).index.tolist()

    print("\nTop 30 feature:")

    print(selected_features)

    X_train = X_train[selected_features]

    X_test = X_test[selected_features]

    return X_train, X_test, selected_features

        
        
def show_dataset_info(df):
    print(df["Label"].value_counts())
    print(df["BinaryLabel"].value_counts())
    print(df.columns)
        
        
        

def preprocess_dataset():
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nessun CSV trovato nella cartella: {data_dir}")

    # Se cambia uno dei CSV, la cache viene ricalcolata automaticamente.
    source_signature = [
        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
        for path in csv_files
    ]
    if cache_file.exists():
        cached = joblib.load(cache_file)
        if cached["source_signature"] == source_signature:
            print(f"Carico il dataset preprocessato da {cache_file}")
            return cached["dataset"]

    df = load_dataset(data_dir)
    df = clean_dataset(df)
    df = prepare_label(df)
    show_dataset_info(df)
    X_train, X_test, y_train, y_test = split_dataset(df)
    X_train, X_test, selected_features = select_top_features(

    X_train,

    X_test,

    y_train

)
    result = (X_train, X_test, y_train, y_test, selected_features)
    joblib.dump(
        {"source_signature": source_signature, "dataset": result},
        cache_file,
        compress=0,
    )
    print(f"Dataset preprocessato salvato in {cache_file}")
    return result



if __name__ == "__main__":
    dataset = preprocess_dataset()
