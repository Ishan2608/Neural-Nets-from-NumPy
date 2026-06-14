"""
network.py
----------
Core neural network implementation: activations, loss functions,
forward/backward passes, parameter initialization, training, and evaluation.
All logic extracted and adapted from ANN_Core_Implementation.ipynb.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# 1. DATASETS
# ─────────────────────────────────────────────

def load_regression_data():
    """
    California Housing dataset.
    Returns processed (standardised) train/test splits plus raw DataFrames.
    """
    california_housing = fetch_california_housing()
    df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    df["target"] = california_housing.target

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert to NumPy
    X_train_np = X_train.to_numpy()
    X_test_np  = X_test.to_numpy()
    y_train_np = y_train.to_numpy()
    y_test_np  = y_test.to_numpy()

    # Standardise features on training statistics
    X_mean = np.mean(X_train_np, axis=0)
    X_std  = np.std(X_train_np,  axis=0)
    X_train_proc = (X_train_np - X_mean) / (X_std + 1e-8)
    X_test_proc  = (X_test_np  - X_mean) / (X_std + 1e-8)

    # Standardise target
    y_mean = np.mean(y_train_np)
    y_std  = np.std(y_train_np)
    y_train_proc = (y_train_np - y_mean) / (y_std + 1e-8)
    y_test_proc  = (y_test_np  - y_mean) / (y_std + 1e-8)

    scaling = {"X_mean": X_mean, "X_std": X_std, "y_mean": y_mean, "y_std": y_std}

    return (
        X_train_proc, X_test_proc,
        y_train_proc, y_test_proc,
        y_train_np,   y_test_np,
        df, scaling
    )


def load_classification_data():
    """
    Breast Cancer dataset.
    Returns processed (standardised) train/test splits plus raw DataFrame.
    """
    breast_cancer = load_breast_cancer()
    df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    df["target"] = breast_cancer.target

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_np = X_train.to_numpy()
    X_test_np  = X_test.to_numpy()
    y_train_np = y_train.to_numpy()
    y_test_np  = y_test.to_numpy()

    # Standardise features
    X_mean = np.mean(X_train_np, axis=0)
    X_std  = np.std(X_train_np,  axis=0)
    X_train_proc = (X_train_np - X_mean) / (X_std + 1e-8)
    X_test_proc  = (X_test_np  - X_mean) / (X_std + 1e-8)

    # Targets already 0/1 – just reshape
    y_train_proc = y_train_np.reshape(-1, 1)
    y_test_proc  = y_test_np.reshape(-1, 1)

    return (
        X_train_proc, X_test_proc,
        y_train_proc, y_test_proc,
        y_train_np,   y_test_np,
        df
    )


# ─────────────────────────────────────────────
# 2. ACTIVATION FUNCTIONS
# ─────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)


# ─────────────────────────────────────────────
# 3. LOSS FUNCTIONS
# ─────────────────────────────────────────────

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def binary_cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# ─────────────────────────────────────────────
# 4. PARAMETER INITIALISATION  (He initialisation)
# ─────────────────────────────────────────────

def initialize_parameters(layers):
    """
    He initialisation – suited for ReLU hidden layers.
    layers : list of neuron counts per layer, e.g. [8, 16, 32, 1]
    """
    weights, biases = [], []
    for i in range(1, len(layers)):
        fan_in = layers[i - 1]
        std = np.sqrt(2.0 / fan_in)
        w = np.random.randn(fan_in, layers[i]) * std
        b = np.zeros((1, layers[i]))
        weights.append(w)
        biases.append(b)
    return weights, biases


# ─────────────────────────────────────────────
# 5. FORWARD PASS
# ─────────────────────────────────────────────

def forward(X, weights, biases, activations):
    """
    Propagates input X through every layer.
    Returns the final output and a cache list (Z, A per layer).
    """
    A = X
    caches = []
    for W, b, act in zip(weights, biases, activations):
        Z = np.dot(A, W) + b
        A = act(Z)
        caches.append({"Z": Z, "A": A})
    return A, caches


# ─────────────────────────────────────────────
# 6. BACKWARD PASS
# ─────────────────────────────────────────────

def backward(X, y_true, y_pred, caches, weights, activations_derivative, problem_type):
    """
    Computes gradients via backpropagation.
    Returns a dict of dW and db for every layer.
    """
    gradients   = {}
    num_layers  = len(weights)
    n_samples   = X.shape[0]

    # ── Output layer gradient ──────────────────
    A_out = caches[num_layers - 1]["A"]
    Z_out = caches[num_layers - 1]["Z"]

    if problem_type == "regression":
        dL_dZ_out = (A_out - y_true) * (2 / n_samples)
    else:   # classification – BCE + sigmoid combined gradient
        dL_dZ_out = (A_out - y_true) / n_samples

    A_prev = caches[num_layers - 2]["A"] if num_layers > 1 else X
    gradients[f"dW{num_layers}"] = np.dot(A_prev.T, dL_dZ_out)
    gradients[f"db{num_layers}"] = np.sum(dL_dZ_out, axis=0, keepdims=True)

    # ── Hidden layers (reverse) ────────────────
    for i in range(num_layers - 2, -1, -1):
        Z_current  = caches[i]["Z"]
        W_next     = weights[i + 1]
        dL_dA      = np.dot(dL_dZ_out, W_next.T)
        dL_dZ_current = dL_dA * activations_derivative[i](Z_current)

        A_prev_layer = caches[i - 1]["A"] if i > 0 else X
        gradients[f"dW{i + 1}"] = np.dot(A_prev_layer.T, dL_dZ_current)
        gradients[f"db{i + 1}"] = np.sum(dL_dZ_current, axis=0, keepdims=True)

        dL_dZ_out = dL_dZ_current

    return gradients


# ─────────────────────────────────────────────
# 7. PARAMETER UPDATE  (Gradient Descent)
# ─────────────────────────────────────────────

def update_parameters(weights, biases, gradients, learning_rate):
    for i in range(len(weights)):
        weights[i] -= learning_rate * gradients[f"dW{i + 1}"]
        biases[i]  -= learning_rate * gradients[f"db{i + 1}"]
    return weights, biases


# ─────────────────────────────────────────────
# 8. TRAINING LOOP
# ─────────────────────────────────────────────

def train_network(
    X, y,
    layers,
    learning_rate=0.001,
    epochs=500,
    problem_type="regression",
    progress_callback=None,
):
    """
    Full training loop.

    Parameters
    ----------
    X, y           : preprocessed NumPy arrays
    layers         : list with neuron counts including input and output
    learning_rate  : gradient-descent step size
    epochs         : number of full passes over the data
    problem_type   : 'regression' or 'classification'
    progress_callback : optional callable(epoch, total, loss) for live updates

    Returns
    -------
    weights, biases, loss_history  (loss recorded every epoch)
    """
    # Build activation lists: ReLU for hidden layers, linear/sigmoid for output
    if problem_type == "regression":
        activations       = [relu        for _ in range(len(layers) - 2)] + [linear]
        activations_derivs= [relu_derivative for _ in range(len(layers) - 2)] + [linear_derivative]
    else:
        activations       = [relu        for _ in range(len(layers) - 2)] + [sigmoid]
        activations_derivs= [relu_derivative for _ in range(len(layers) - 2)] + [sigmoid_derivative]

    weights, biases = initialize_parameters(layers)
    y = y.reshape(-1, 1)
    loss_history = []

    for epoch in range(1, epochs + 1):
        y_pred, caches = forward(X, weights, biases, activations)

        if problem_type == "regression":
            loss = mse_loss(y, y_pred)
        else:
            loss = binary_cross_entropy_loss(y, y_pred)

        gradients = backward(X, y, y_pred, caches, weights, activations_derivs, problem_type)
        weights, biases = update_parameters(weights, biases, gradients, learning_rate)

        loss_history.append(float(loss))

        if progress_callback:
            progress_callback(epoch, epochs, float(loss))

    return weights, biases, loss_history, activations


# ─────────────────────────────────────────────
# 9. PREDICTION & EVALUATION
# ─────────────────────────────────────────────

def predict(X, weights, biases, activations):
    y_pred, _ = forward(X, weights, biases, activations)
    return y_pred


def evaluate_regression(weights, biases, activations, X_test, y_test_proc, y_test_raw, scaling):
    """
    Returns MSE on standardised scale and MAE/R² on original scale.
    """
    preds_proc = predict(X_test, weights, biases, activations).flatten()
    mse = float(mse_loss(y_test_proc.reshape(-1), preds_proc))

    # Revert standardisation
    y_mean, y_std = scaling["y_mean"], scaling["y_std"]
    preds_raw = preds_proc * y_std + y_mean

    mae = float(np.mean(np.abs(y_test_raw - preds_raw)))
    ss_res = np.sum((y_test_raw - preds_raw) ** 2)
    ss_tot = np.sum((y_test_raw - np.mean(y_test_raw)) ** 2)
    r2  = float(1 - ss_res / ss_tot)

    return {"mse": mse, "mae": mae, "r2": r2, "preds_raw": preds_raw, "y_test_raw": y_test_raw}


def evaluate_classification(weights, biases, activations, X_test, y_test_raw, threshold=0.5):
    """
    Returns accuracy, and arrays of probs and binary predictions.
    """
    probs        = predict(X_test, weights, biases, activations).flatten()
    preds_binary = (probs >= threshold).astype(int)
    accuracy     = float(np.mean(preds_binary == y_test_raw.flatten()))
    return {"accuracy": accuracy, "probs": probs, "preds_binary": preds_binary}
