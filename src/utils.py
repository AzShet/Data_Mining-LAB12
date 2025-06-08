
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from category_encoders import OneHotEncoder
from typing import Tuple, List

def load_data(path: str) -> pl.DataFrame:
    """
    Carga el dataset desde un archivo CSV utilizando polars.

    Args:
        path (str): Ruta al archivo CSV.

    Returns:
        pl.DataFrame: DataFrame con los datos cargados.
    """
    return pl.read_csv(path)

def check_missing_values(df: pl.DataFrame) -> pl.DataFrame:
    """
    Verifica la presencia de valores faltantes en el DataFrame.

    Args:
        df (pl.DataFrame): DataFrame a verificar.

    Returns:
        pl.DataFrame: DataFrame con el conteo de valores faltantes por columna.
    """
    return df.null_count()

def remove_outliers(df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
    """
    Elimina outliers utilizando el método del rango intercuartílico (IQR).

    Args:
        df (pl.DataFrame): DataFrame original.
        columns (List[str]): Lista de nombres de columnas numéricas.

    Returns:
        pl.DataFrame: DataFrame sin outliers.
    """
    for col in columns:
        q1 = df.select(pl.col(col).quantile(0.25)).to_series()[0]
        q3 = df.select(pl.col(col).quantile(0.75)).to_series()[0]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df.filter((pl.col(col) >= lower_bound) & (pl.col(col) <= upper_bound))
    return df

def encode_categorical(df: pl.DataFrame, categorical_cols: List[str]) -> Tuple[pl.DataFrame, List[str]]:
    """
    Codifica variables categóricas utilizando one-hot encoding con Polars.

    Args:
        df (pl.DataFrame): DataFrame original.
        categorical_cols (List[str]): Columnas categóricas a codificar.

    Returns:
        Tuple[pl.DataFrame, List[str]]: DataFrame codificado y nombres de las columnas resultantes.
    """
    df_cat = df.select(categorical_cols)
    df_encoded = df_cat.to_dummies()
    df_non_cat = df.drop(categorical_cols)
    final_df = df_non_cat.hstack(df_encoded)
    return final_df, final_df.columns


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide los datos en conjuntos de entrenamiento y prueba.

    Args:
        X (np.ndarray): Variables independientes.
        y (np.ndarray): Variable objetivo.
        test_size (float, optional): Proporción del conjunto de prueba. Default es 0.2.
        random_state (int, optional): Semilla para reproducibilidad. Default es 42.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Conjuntos de entrenamiento y prueba para X e y.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def train_and_evaluate_model(model, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> float:
    """
    Entrena y evalúa un modelo de clasificación.

    Args:
        model: Modelo de clasificación.
        X_train (np.ndarray): Características de entrenamiento.
        X_test (np.ndarray): Características de prueba.
        y_train (np.ndarray): Etiquetas de entrenamiento.
        y_test (np.ndarray): Etiquetas de prueba.

    Returns:
        float: Precisión del modelo en el conjunto de prueba.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return acc
