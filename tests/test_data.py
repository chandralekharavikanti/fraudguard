import pandas as pd
import pytest
from fraudguard.data import FEATURES, load_data

def test_missing_real_data_does_not_generate_fake_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "creditcard.csv")

def test_valid_schema(tmp_path):
    path = tmp_path / "creditcard.csv"
    rows = [{**{k: 0.0 for k in FEATURES}, "Class": label} for label in (0, 1)]
    pd.DataFrame(rows).to_csv(path, index=False)
    assert len(load_data(path)) == 2

def test_rejects_missing_features(tmp_path):
    path = tmp_path / "creditcard.csv"
    pd.DataFrame({"Class": [0, 1]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_data(path)
