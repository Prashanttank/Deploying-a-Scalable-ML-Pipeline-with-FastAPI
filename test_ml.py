import pytest
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data
import pandas as pd
import numpy as np

# Load the dataset
data_path = "data/census.csv"
data = pd.read_csv(data_path)
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Split the data for testing purposes
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# Process the data
X_train, y_train, encoder, lb = process_data(
    train_data, categorical_features=cat_features, label="salary", training=True
)
X_test, y_test, _, _ = process_data(
    test_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Test to verify that model training works correctly
def test_train_model():
    model = train_model(X_train, y_train)
    assert model is not None, "Model training failed - model is None"

# Test to check that compute_model_metrics returns expected metrics
def test_compute_model_metrics():
    # Use inference to generate predictions for testing metrics
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert 0 <= precision <= 1, "Precision out of bounds"
    assert 0 <= recall <= 1, "Recall out of bounds"
    assert 0 <= fbeta <= 1, "F1 Score out of bounds"

# Test to check that inference generates predictions in the expected format
def test_inference():
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray), "Inference output is not a numpy array"
    assert len(preds) == len(X_test), "Inference output length mismatch with test data"
