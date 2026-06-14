"""
main.py
-------
Gradio UI for the custom ANN implemented in network.py.
Run with:  python main.py
"""

import time
import threading
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gradio as gr

from network import (
    load_regression_data,
    load_classification_data,
    train_network,
    evaluate_regression,
    evaluate_classification,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE  (loaded once, reused across runs)
# ─────────────────────────────────────────────────────────────────────────────
_reg_data  = None
_clf_data  = None

def get_regression_data():
    global _reg_data
    if _reg_data is None:
        _reg_data = load_regression_data()
    return _reg_data

def get_classification_data():
    global _clf_data
    if _clf_data is None:
        _clf_data = load_classification_data()
    return _clf_data


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG   = "#0f1117"
CARD_BG   = "#1a1d27"
ACCENT    = "#6366f1"     # indigo
GREEN     = "#22c55e"
RED_COLOR = "#ef4444"
TEXT      = "#e2e8f0"
SUBTEXT   = "#94a3b8"
GRID_CLR  = "#2d3148"

def _style_fig(fig):
    fig.patch.set_facecolor(DARK_BG)
    for ax in fig.get_axes():
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=SUBTEXT, labelsize=8)
        ax.xaxis.label.set_color(SUBTEXT)
        ax.yaxis.label.set_color(SUBTEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_CLR)
        ax.grid(True, color=GRID_CLR, linewidth=0.5, linestyle="--")
    return fig


def parse_hidden_layers(hidden_str):
    """'64, 32, 16'  →  [64, 32, 16]"""
    try:
        return [int(x.strip()) for x in hidden_str.split(",") if x.strip()]
    except Exception:
        return [64, 32]


def dataset_preview(problem_type):
    if problem_type == "Regression":
        *_, df, _ = get_regression_data()
        info = (
            "**California Housing** — 20 640 rows, 8 features.  "
            "Target: median house value (in $100k)."
        )
    else:
        *_, df = get_classification_data()
        info = (
            "**Breast Cancer Wisconsin** — 569 rows, 30 features.  "
            "Target: 0 = malignant, 1 = benign."
        )
    return info, df.head(10)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def run_training(problem_type, hidden_layers_str, learning_rate, epochs, progress=gr.Progress()):
    """
    Trains the network and returns all results / plots.
    """
    hidden = parse_hidden_layers(hidden_layers_str)

    # ── Load data ────────────────────────────────
    if problem_type == "Regression":
        (X_train, X_test, y_train, y_test,
         y_train_raw, y_test_raw, _, scaling) = get_regression_data()
        input_dim    = X_train.shape[1]
        output_dim   = 1
        p_type       = "regression"
        loss_label   = "MSE Loss"
    else:
        (X_train, X_test, y_train, y_test,
         y_train_raw, y_test_raw, _) = get_classification_data()
        input_dim  = X_train.shape[1]
        output_dim = 1
        p_type     = "classification"
        loss_label = "Binary Cross-Entropy Loss"
        scaling    = None

    layers = [input_dim] + hidden + [output_dim]

    # ── Progress tracking ─────────────────────────
    loss_log   = []
    epoch_log  = []
    lock       = threading.Lock()

    def callback(epoch, total, loss):
        with lock:
            loss_log.append(loss)
            epoch_log.append(epoch)
        pct = epoch / total
        progress(pct, desc=f"Epoch {epoch}/{total}  |  Loss: {loss:.5f}")

    # ── Train ─────────────────────────────────────
    t0 = time.time()
    Ws, Bs, loss_history, activations = train_network(
        X_train, y_train,
        layers=layers,
        learning_rate=learning_rate,
        epochs=int(epochs),
        problem_type=p_type,
        progress_callback=callback,
    )
    elapsed = time.time() - t0

    # ── Evaluate ──────────────────────────────────
    if p_type == "regression":
        eval_res = evaluate_regression(
            Ws, Bs, activations, X_test,
            y_test, y_test_raw, scaling
        )
    else:
        eval_res = evaluate_classification(
            Ws, Bs, activations, X_test, y_test_raw
        )

    # ── Build plots ───────────────────────────────
    fig = _make_plots(loss_history, eval_res, p_type, loss_label, layers)

    # ── Metrics text ──────────────────────────────
    arch_str = " → ".join(str(n) for n in layers)
    metrics  = _metrics_markdown(eval_res, p_type, elapsed, arch_str, layers)

    return fig, metrics


def _make_plots(loss_history, eval_res, p_type, loss_label, layers):
    if p_type == "regression":
        fig = plt.figure(figsize=(14, 4.5), facecolor=DARK_BG)
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

        # 1) Loss curve
        ax0 = fig.add_subplot(gs[0])
        ax0.plot(loss_history, color=ACCENT, linewidth=1.6)
        ax0.set_title("Training Loss Curve")
        ax0.set_xlabel("Epoch")
        ax0.set_ylabel(loss_label)

        # 2) Actual vs Predicted scatter (test set, up to 300 pts)
        ax1 = fig.add_subplot(gs[1])
        y_raw   = eval_res["y_test_raw"][:300]
        p_raw   = eval_res["preds_raw"][:300]
        ax1.scatter(y_raw, p_raw, alpha=0.45, s=14, color=ACCENT, edgecolors="none")
        mn, mx = min(y_raw.min(), p_raw.min()), max(y_raw.max(), p_raw.max())
        ax1.plot([mn, mx], [mn, mx], color=GREEN, linewidth=1.2, linestyle="--", label="Perfect fit")
        ax1.set_title("Actual vs Predicted (Test)")
        ax1.set_xlabel("Actual")
        ax1.set_ylabel("Predicted")
        ax1.legend(fontsize=7, facecolor=CARD_BG, labelcolor=TEXT)

        # 3) Residuals
        ax2 = fig.add_subplot(gs[2])
        residuals = y_raw - p_raw
        ax2.hist(residuals, bins=30, color=ACCENT, alpha=0.75, edgecolor=DARK_BG)
        ax2.axvline(0, color=GREEN, linewidth=1.2, linestyle="--")
        ax2.set_title("Residual Distribution (Test)")
        ax2.set_xlabel("Residual  (Actual − Predicted)")
        ax2.set_ylabel("Count")

    else:   # classification
        fig = plt.figure(figsize=(14, 4.5), facecolor=DARK_BG)
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

        # 1) Loss curve
        ax0 = fig.add_subplot(gs[0])
        ax0.plot(loss_history, color=ACCENT, linewidth=1.6)
        ax0.set_title("Training Loss Curve")
        ax0.set_xlabel("Epoch")
        ax0.set_ylabel(loss_label)

        # 2) Probability distribution
        ax1 = fig.add_subplot(gs[1])
        probs = eval_res["probs"]
        y_raw = eval_res["preds_binary"]
        ax1.hist(probs[y_raw == 1], bins=25, alpha=0.65, color=GREEN,
                 label="Predicted Benign",  edgecolor=DARK_BG)
        ax1.hist(probs[y_raw == 0], bins=25, alpha=0.65, color=RED_COLOR,
                 label="Predicted Malignant", edgecolor=DARK_BG)
        ax1.axvline(0.5, color=TEXT, linewidth=1.2, linestyle="--", label="Threshold = 0.5")
        ax1.set_title("Predicted Probability Distribution")
        ax1.set_xlabel("Sigmoid Output (Probability)")
        ax1.set_ylabel("Count")
        ax1.legend(fontsize=7, facecolor=CARD_BG, labelcolor=TEXT)

        # 3) Confusion matrix
        ax2 = fig.add_subplot(gs[2])
        y_true_raw = eval_res["preds_binary"]   # reuse preds vs actuals trick below
        # We need actuals — pass them via eval_res
        y_actual   = eval_res.get("y_actual", eval_res["preds_binary"])  # fallback safe
        preds_b    = eval_res["preds_binary"]

        # We stored y_test_raw inside evaluate_classification via a wrapper below
        _y_true = eval_res.get("y_true", preds_b)

        tp = int(np.sum((_y_true == 1) & (preds_b == 1)))
        tn = int(np.sum((_y_true == 0) & (preds_b == 0)))
        fp = int(np.sum((_y_true == 0) & (preds_b == 1)))
        fn = int(np.sum((_y_true == 1) & (preds_b == 0)))
        cm = np.array([[tn, fp], [fn, tp]])

        im = ax2.imshow(cm, cmap="Blues")
        ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
        ax2.set_xticklabels(["Pred Malignant", "Pred Benign"], color=TEXT, fontsize=7.5)
        ax2.set_yticklabels(["Actual Malignant", "Actual Benign"], color=TEXT, fontsize=7.5)
        ax2.set_title("Confusion Matrix (Test)")
        for (r, c), val in np.ndenumerate(cm):
            ax2.text(c, r, str(val), ha="center", va="center",
                     fontsize=14, color=TEXT, fontweight="bold")
        plt.colorbar(im, ax=ax2, fraction=0.04)

    _style_fig(fig)
    plt.tight_layout(pad=2.0)
    return fig


def _metrics_markdown(eval_res, p_type, elapsed, arch_str, layers):
    n_params = 0
    for i in range(1, len(layers)):
        n_params += layers[i - 1] * layers[i] + layers[i]

    header = f"""
### Training Complete  ({elapsed:.1f}s)

| Detail | Value |
|---|---|
| Architecture | `{arch_str}` |
| Total Parameters | {n_params:,} |
| Training Time | {elapsed:.2f} s |
"""
    if p_type == "regression":
        r = eval_res
        body = f"""
#### Test-Set Metrics

| Metric | Value |
|---|---|
| MSE (standardised) | {r['mse']:.5f} |
| MAE (original scale) | {r['mae']:.4f} |
| R² Score | {r['r2']:.4f} |
"""
    else:
        acc = eval_res["accuracy"]
        body = f"""
#### Test-Set Metrics

| Metric | Value |
|---|---|
| Accuracy | **{acc * 100:.2f}%** |
"""
    return header + body


# ─────────────────────────────────────────────────────────────────────────────
# WRAPPER  – injects y_true into eval_res for the confusion matrix
# ─────────────────────────────────────────────────────────────────────────────
_orig_evaluate_classification = evaluate_classification

def _evaluate_classification_with_truth(weights, biases, activations, X_test, y_test_raw, threshold=0.5):
    res = _orig_evaluate_classification(weights, biases, activations, X_test, y_test_raw, threshold)
    res["y_true"] = y_test_raw.flatten()
    return res

import network as _net_module
_net_module.evaluate_classification = _evaluate_classification_with_truth


# Patch run_training to use the patched evaluate
def run_training_patched(problem_type, hidden_layers_str, learning_rate, epochs, progress=gr.Progress()):
    hidden = parse_hidden_layers(hidden_layers_str)

    if problem_type == "Regression":
        (X_train, X_test, y_train, y_test,
         y_train_raw, y_test_raw, _, scaling) = get_regression_data()
        input_dim  = X_train.shape[1]
        output_dim = 1
        p_type     = "regression"
        loss_label = "MSE Loss"
    else:
        (X_train, X_test, y_train, y_test,
         y_train_raw, y_test_raw, _) = get_classification_data()
        input_dim  = X_train.shape[1]
        output_dim = 1
        p_type     = "classification"
        loss_label = "Binary Cross-Entropy Loss"
        scaling    = None

    layers = [input_dim] + hidden + [output_dim]

    loss_log  = []
    lock      = threading.Lock()

    def callback(epoch, total, loss):
        with lock:
            loss_log.append(loss)
        progress(epoch / total, desc=f"Epoch {epoch}/{total}  |  Loss: {loss:.5f}")

    t0 = time.time()
    Ws, Bs, loss_history, activations = train_network(
        X_train, y_train,
        layers=layers,
        learning_rate=learning_rate,
        epochs=int(epochs),
        problem_type=p_type,
        progress_callback=callback,
    )
    elapsed = time.time() - t0

    if p_type == "regression":
        eval_res = evaluate_regression(Ws, Bs, activations, X_test, y_test, y_test_raw, scaling)
    else:
        eval_res = _evaluate_classification_with_truth(Ws, Bs, activations, X_test, y_test_raw)

    fig     = _make_plots(loss_history, eval_res, p_type, loss_label, layers)
    metrics = _metrics_markdown(eval_res, p_type, elapsed, " → ".join(str(n) for n in layers), layers)

    return fig, metrics


# ─────────────────────────────────────────────────────────────────────────────
# CSS  (dark, clean, minimal)
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
/* ── page background ── */
body, .gradio-container { background:#0f1117 !important; color:#e2e8f0 !important; font-family:'Inter',sans-serif; }

/* ── cards ── */
.gr-box, .gr-form, .gr-panel, .gr-block, .gr-group,
.block.svelte-1ipelgc { background:#1a1d27 !important; border:1px solid #2d3148 !important; border-radius:12px !important; }

/* ── labels & text ── */
label span, .gr-label, .svelte-1ipelgc label { color:#94a3b8 !important; font-size:0.78rem !important; letter-spacing:0.04em; text-transform:uppercase; }

/* ── inputs ── */
input, textarea, select,
.gr-input input, .gr-input textarea {
    background:#0f1117 !important; color:#e2e8f0 !important;
    border:1px solid #2d3148 !important; border-radius:8px !important;
}
input:focus, textarea:focus { border-color:#6366f1 !important; outline:none !important; box-shadow:0 0 0 2px rgba(99,102,241,.25) !important; }

/* ── sliders ── */
.gr-slider input[type=range] { accent-color:#6366f1; }

/* ── primary button ── */
.gr-button-primary, button.primary {
    background:linear-gradient(135deg,#6366f1,#818cf8) !important;
    color:#fff !important; border:none !important; border-radius:8px !important;
    font-weight:600; letter-spacing:0.03em; transition:opacity .2s;
}
.gr-button-primary:hover, button.primary:hover { opacity:.88 !important; }

/* ── secondary button ── */
.gr-button-secondary, button.secondary {
    background:#1a1d27 !important; color:#94a3b8 !important;
    border:1px solid #2d3148 !important; border-radius:8px !important;
}

/* ── tabs ── */
.tab-nav button { color:#94a3b8 !important; border-bottom:2px solid transparent; }
.tab-nav button.selected { color:#6366f1 !important; border-bottom:2px solid #6366f1 !important; }

/* ── markdown metrics table ── */
.gr-markdown table { border-collapse:collapse; width:100%; }
.gr-markdown th { background:#2d3148; color:#94a3b8; font-size:.75rem; text-transform:uppercase; padding:6px 10px; }
.gr-markdown td { padding:6px 10px; border-bottom:1px solid #2d3148; color:#e2e8f0; font-size:.85rem; }
.gr-markdown code { background:#0f1117; padding:1px 5px; border-radius:4px; font-size:.8rem; color:#818cf8; }

/* ── progress bar ── */
.progress-bar { background:#6366f1 !important; }

/* ── dataframe ── */
.gr-dataframe table th { background:#2d3148 !important; color:#94a3b8 !important; }
.gr-dataframe table td { color:#e2e8f0 !important; font-size:.78rem !important; }
"""

# ─────────────────────────────────────────────────────────────────────────────
# BUILD UI
# ─────────────────────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(css=CSS, title="ANN Builder") as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.Markdown(
            """
            # ANN Builder
            Build, train, and evaluate a custom Artificial Neural Network — from scratch, no frameworks.
            Choose a task, configure the architecture, and hit **Train**.
            """
        )

        with gr.Tabs():

            # ════════════════════════════════════════════════════════════════
            # TAB 1  –  Train
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("Train Network"):

                with gr.Row():

                    # ── Left column: controls ────────────────────────────
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("### Configuration")

                        problem_type = gr.Dropdown(
                            choices=["Regression", "Classification"],
                            value="Regression",
                            label="Task",
                            info="Regression → California Housing  |  Classification → Breast Cancer",
                        )

                        hidden_layers = gr.Textbox(
                            value="64, 32, 16",
                            label="Hidden Layer Sizes",
                            info="Comma-separated neuron counts, e.g. '128, 64, 32'",
                            placeholder="64, 32, 16",
                        )

                        learning_rate = gr.Slider(
                            minimum=0.0001, maximum=0.01, step=0.0001,
                            value=0.001, label="Learning Rate",
                        )

                        epochs = gr.Slider(
                            minimum=100, maximum=2000, step=100,
                            value=500, label="Epochs",
                        )

                        gr.Markdown(
                            """
                            **Notes**
                            - Hidden layers use **ReLU** activation.
                            - Regression output: **linear** + MSE loss.
                            - Classification output: **sigmoid** + Binary Cross-Entropy.
                            - Weights use **He initialisation**.
                            - Optimiser: **vanilla gradient descent**.
                            """
                        )

                        train_btn = gr.Button("Train", variant="primary", size="lg")

                    # ── Right column: outputs ────────────────────────────
                    with gr.Column(scale=2):
                        gr.Markdown("### Results")
                        plot_out   = gr.Plot(label="Training & Evaluation Plots")
                        metric_out = gr.Markdown("*Metrics will appear here after training.*")

                train_btn.click(
                    fn=run_training_patched,
                    inputs=[problem_type, hidden_layers, learning_rate, epochs],
                    outputs=[plot_out, metric_out],
                )

            # ════════════════════════════════════════════════════════════════
            # TAB 2  –  Explore Data
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("Explore Data"):
                gr.Markdown("### Dataset Preview")
                gr.Markdown(
                    "Select a task to inspect the first 10 rows and summary statistics of the dataset "
                    "that will be used for training."
                )

                with gr.Row():
                    ds_choice = gr.Dropdown(
                        choices=["Regression", "Classification"],
                        value="Regression",
                        label="Dataset",
                        scale=1,
                    )
                    load_btn = gr.Button("Load", variant="secondary", scale=0)

                ds_info = gr.Markdown()
                ds_table = gr.Dataframe(
                    label="First 10 Rows",
                    wrap=True,
                    interactive=False,
                )

                load_btn.click(
                    fn=dataset_preview,
                    inputs=[ds_choice],
                    outputs=[ds_info, ds_table],
                )

                # Auto-load on dropdown change
                ds_choice.change(
                    fn=dataset_preview,
                    inputs=[ds_choice],
                    outputs=[ds_info, ds_table],
                )

                # Default load
                demo.load(
                    fn=dataset_preview,
                    inputs=[ds_choice],
                    outputs=[ds_info, ds_table],
                )

            # ════════════════════════════════════════════════════════════════
            # TAB 3  –  How It Works
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("How It Works"):
                gr.Markdown(
                    """
                    ### How the Network is Built (Zero Frameworks)

                    Everything here is implemented using **NumPy only** — no TensorFlow, no PyTorch.

                    **Forward Pass**
                    Each layer computes `Z = X · W + b`, then applies an activation function to get `A`.
                    The output of one layer becomes the input of the next.

                    **Loss Functions**
                    - Regression → Mean Squared Error (MSE): average of squared differences between actual and predicted values.
                    - Classification → Binary Cross-Entropy: measures how far the predicted probability is from the true label (0 or 1).

                    **Backward Pass (Backpropagation)**
                    The gradient of the loss is computed with respect to every weight and bias using the chain rule, working backwards from the output layer to the input layer.

                    **Weight Update (Gradient Descent)**
                    `W = W - α · dL/dW` where α is the learning rate.
                    Each weight is nudged in the direction that reduces the loss.

                    **Initialisation**
                    He initialisation: `W ~ N(0, √(2/fan_in))`.
                    This keeps the signal from exploding or vanishing through many ReLU layers.

                    **Architecture**
                    - Input neurons = number of features in the dataset.
                    - Hidden layers = your choice (ReLU activation throughout).
                    - Output neuron = 1 (linear for regression, sigmoid for classification).
                    """
                )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_ui()
    app.launch(show_error=True)
