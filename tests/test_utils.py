# tests
import pytest
import polars as pl
import numpy as np
from sklearn.naive_bayes import GaussianNB
from src.utils import (
    load_data,
    check_missing_values,
    remove_outliers,
    encode_categorical,
    split_data,
    train_and_evaluate_model
)

def test_load_data(tmp_path):
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("a,b\n1,2\n3,4")
    df = load_data(str(csv_path))
    assert df.shape == (2, 2)
    assert df.columns == ["a", "b"]

def test_check_missing_values():
    df = pl.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
    result = check_missing_values(df)
    assert result.select(pl.col("a")).item() == 1
    assert result.select(pl.col("b")).item() == 1


def test_remove_outliers():
    df = pl.DataFrame({"a": [1, 2, 3, 100]})
    cleaned = remove_outliers(df, ["a"])
    assert 100 not in cleaned["a"].to_list()
    assert len(cleaned) == 3

def test_encode_categorical():
    df = pl.DataFrame({"color": ["red", "blue", "red"], "size": ["S", "M", "L"], "value": [10, 20, 30]})
    encoded_df, feature_names = encode_categorical(df, ["color", "size"])
    assert "value" in feature_names
    assert any("color_red" in col or "color_blue" in col for col in feature_names)
    assert encoded_df.shape[0] == 3

def test_split_data():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.5, random_state=0)
    assert len(X_train) == 2
    assert len(X_test) == 2
    assert sorted(np.concatenate([y_train, y_test])) == [0, 0, 1, 1]

def test_train_and_evaluate_model(capsys):
    X = np.array([[1, 2], [1, 3], [4, 5], [6, 7]])
    y = np.array([0, 0, 1, 1])
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.5, random_state=0)
    acc = train_and_evaluate_model(GaussianNB(), X_train, X_test, y_train, y_test)
    captured = capsys.readouterr()
    assert "Accuracy:" in captured.out
    assert 0 <= acc <= 1
