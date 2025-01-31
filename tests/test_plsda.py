import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import pytest
from momics_ad.stats import pls


@pytest.fixture
def synthetic_data():
    """
    Fixture to set up small synthetic test data for X and y.
    X: Predictor variables
    y: Outcome variable
    """
    np.random.seed(42)
    X = pd.DataFrame(np.random.random((30, 5)), columns=[f"feature_{i}" for i in range(5)])
    y = pd.Series(np.random.choice([0, 1], size=30))
    return X, y


def test_plsda_doubleCV(synthetic_data):
    """
    Test the plsda_doubleCV function with synthetic data.
    """
    X, y = synthetic_data
    result = pls.plsda_doubleCV(X, y, cv1_splits=3, cv2_splits=2, n_repeats=2, max_components=5)

    # Validate output format
    assert "models" in result
    assert "table" in result
    assert isinstance(result["models"], list)
    assert isinstance(result["table"], pd.DataFrame)

    # Check table shape and values
    assert result["table"].shape[1] == 3  # Expect columns: "rep", "LV", "AUROC"
    assert (result["table"]["AUROC"] >= 0).all()  # AUROC should be >= 0


def test_plsda_auroc(synthetic_data):
    """
    Test the _plsda_auroc function with known inputs and outputs.
    """
    X, y = synthetic_data
    encoder = OneHotEncoder(sparse_output=False)
    encoded_y = pd.DataFrame(encoder.fit_transform(y.values.reshape(-1, 1)))
    pls_reg = PLSRegression(n_components=2).fit(X, encoded_y)

    # Generate test AUROC
    y_pred = pls_reg.predict(X)
    expected_auroc = roc_auc_score(encoded_y, y_pred)

    test_auroc = pls._plsda_auroc(2, X, encoded_y, X, encoded_y)
    assert pytest.approx(test_auroc, rel=1e-5) == expected_auroc

    # Test return_full=True
    test_result = pls._plsda_auroc(2, X, encoded_y, X, encoded_y, return_full=True)
    assert "model" in test_result
    assert "score" in test_result
    assert pytest.approx(test_result["score"], rel=1e-5) == expected_auroc


def test_calculate_vips(synthetic_data):
    """
    Test the _calculate_vips function for VIP values.
    """
    X, y = synthetic_data
    pls_reg = PLSRegression(n_components=2).fit(X, y)

    # Calculate VIPs
    vips = pls._calculate_vips(pls_reg)
    assert vips.shape[0] == X.shape[1]  # VIPs should match the number of features
    assert np.all(vips >= 0)  # VIP values should be non-negative


def test_edge_cases():
    """
    Test edge cases for invalid or small inputs.
    """
    # Empty dataframe or series
    with pytest.raises(ValueError):
        pls.plsda_doubleCV(pd.DataFrame(), pd.Series())

    # Single row in X and y
    small_X = pd.DataFrame([[1, 2, 3, 4, 5]])
    small_y = pd.Series([0])
    with pytest.raises(ValueError):
        pls.plsda_doubleCV(small_X, small_y)


def test_default_parameters(synthetic_data):
    """
    Test if the default parameters produce consistent results.
    """
    X, y = synthetic_data
    # TODO: max components needs to be set to a small number in this test.
    # Check why
    result = pls.plsda_doubleCV(X, y, max_components=5)

    assert "models" in result
    assert "table" in result

    # Validate AUROC values in result table
    assert (result["table"]["AUROC"] >= 0).all()
