"""Generate clean, README-ready graphs for the Candle-Predictor project."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style setup
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "font.family": "sans-serif",
    "font.size": 11,
})


def generate_architecture_diagram():
    """Generate a visual pipeline/architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Color palette
    colors = {
        "input": "#1f6feb",
        "preprocess": "#238636",
        "model": "#8957e5",
        "output": "#da3633",
        "arrow": "#58a6ff",
    }

    boxes = [
        (0.3, 1.8, 2.0, 1.4, "Raw OHLCV\nCandle Data", colors["input"]),
        (3.0, 1.8, 2.2, 1.4, "Preprocessing\n+ Trend Mask\n+ EWMA Features", colors["preprocess"]),
        (6.0, 2.8, 2.4, 1.4, "Classifier\nTransformer\n(8L, 8H, 320d)", colors["model"]),
        (6.0, 0.8, 2.4, 1.4, "Regressor\nTransformer\n(8L, 8H, 320d)", colors["model"]),
        (9.2, 2.8, 2.4, 1.4, "Trend / No-Trend\nBinary Label", colors["output"]),
        (9.2, 0.8, 2.4, 1.4, "HLC Delta\nPredictions", colors["output"]),
    ]

    for x, y, w, h, text, color in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.12",
            facecolor=color,
            edgecolor="#ffffff",
            linewidth=1.2,
            alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center", fontsize=9,
                fontweight="bold", color="white", linespacing=1.4)

    # Arrows
    arrow_style = dict(arrowstyle="->,head_width=0.3,head_length=0.15",
                       color=colors["arrow"], lw=2)

    # Input -> Preprocess
    ax.annotate("", xy=(3.0, 2.5), xytext=(2.3, 2.5), arrowprops=arrow_style)
    # Preprocess -> Classifier
    ax.annotate("", xy=(6.0, 3.5), xytext=(5.2, 2.8), arrowprops=arrow_style)
    # Preprocess -> Regressor
    ax.annotate("", xy=(6.0, 1.5), xytext=(5.2, 2.2), arrowprops=arrow_style)
    # Classifier -> Binary label
    ax.annotate("", xy=(9.2, 3.5), xytext=(8.4, 3.5), arrowprops=arrow_style)
    # Regressor -> HLC Delta
    ax.annotate("", xy=(9.2, 1.5), xytext=(8.4, 1.5), arrowprops=arrow_style)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "architecture.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Generated architecture.png")


def generate_performance_summary():
    """Generate a bar chart of classifier performance metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Left: accuracy bar ---
    stats = {
        "Accuracy": 76.78,
        "FP Rate": 13.33,
        "FN Rate": 37.50,
    }
    colors = ["#238636", "#da3633", "#d29922"]
    bars = axes[0].bar(stats.keys(), stats.values(), color=colors, width=0.55, edgecolor="#30363d")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_title("Classifier Metrics", fontweight="bold", fontsize=13)
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, stats.values()):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     f"{val:.1f}%", ha="center", va="bottom", fontweight="bold",
                     fontsize=11, color="#c9d1d9")

    # --- Right: confusion matrix style ---
    cm = np.array([[16, 104], [3, 5]])  # FP=16, TN=104, FN=3, TP=5
    # Recompute to be: [[TP, FP], [FN, TN]]
    # From stats: FP=16, FN=3
    # Accuracy 76.78% on 128-sample batches
    # Let's derive: total predictions in last eval
    # FP=16, FN=3, FPR=0.1333 => FP+TN = 16/0.1333 = 120 => TN=104
    # FNR=0.375 => FN+TP = 3/0.375 = 8 => TP=5
    # Total = 120 + 8 = 128
    cm = np.array([[5, 16], [3, 104]])  # [[TP, FP], [FN, TN]]

    im = axes[1].imshow(cm, cmap="Blues", alpha=0.7)
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(["Trend", "No Trend"])
    axes[1].set_yticklabels(["Trend", "No Trend"])
    axes[1].set_xlabel("Predicted", fontsize=11)
    axes[1].set_ylabel("Actual", fontsize=11)
    axes[1].set_title("Confusion Matrix", fontweight="bold", fontsize=13)

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > 50 else "#c9d1d9"
            axes[1].text(j, i, str(cm[i, j]),
                         ha="center", va="center", fontsize=16,
                         fontweight="bold", color=color)

    fig.tight_layout(pad=2.0)
    fig.savefig(os.path.join(OUTPUT_DIR, "classifier_performance.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Generated classifier_performance.png")


def generate_lr_schedule():
    """Visualize the learning rate schedule (warmup + cosine annealing with warm restarts)."""
    import math

    WARMUP = 9000
    T0 = 10000
    T_MULT = 2
    LR = 6e-6
    total_steps = 80000

    def get_lr_multiplier(step):
        if step < WARMUP:
            return step / max(1, WARMUP)
        s = step - WARMUP
        cycle_len = T0
        cycle_start = 0
        while s >= cycle_start + cycle_len:
            cycle_start += cycle_len
            cycle_len *= T_MULT
        progress = (s - cycle_start) / max(1, cycle_len)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    steps = np.arange(total_steps)
    lr_values = np.array([LR * get_lr_multiplier(s) for s in steps])

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(steps, lr_values * 1e6, color="#58a6ff", linewidth=1.2)
    ax.fill_between(steps, lr_values * 1e6, alpha=0.15, color="#58a6ff")

    ax.axvline(x=WARMUP, color="#da3633", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(WARMUP + 500, LR * 1e6 * 0.95, "Warmup\nends", fontsize=9, color="#da3633")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate (1e-6)")
    ax.set_title("Learning Rate Schedule: Linear Warmup + Cosine Annealing with Warm Restarts",
                  fontweight="bold", fontsize=11)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "lr_schedule.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Generated lr_schedule.png")


def generate_feature_pipeline():
    """Generate a diagram showing the feature engineering pipeline."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # Raw features column
    raw_features = ["open", "high", "low", "close", "volume", "datetime"]
    derived_features = [
        "OHLC diffs",
        "asinh(HLC diffs)",
        "log1p(open, vol)",
        "volume pct change",
        "sin/cos time",
        "EWMA velocity",
        "EWMA acceleration",
        "trend mask (label)",
    ]
    window_features = [
        "Window-normalized OHLC",
        "11 derived features",
        "480-step sliding window",
    ]

    col_x = [0.5, 3.8, 7.8]
    col_titles = ["Raw Input (6)", "Feature Engineering (8)", "Model Input (15d x 480)"]
    col_items = [raw_features, derived_features, window_features]
    col_colors = ["#1f6feb", "#238636", "#8957e5"]

    for cx, title, items, color in zip(col_x, col_titles, col_items, col_colors):
        # Title
        ax.text(cx + 1.2, 3.7, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)
        # Items
        for i, item in enumerate(items):
            y = 3.3 - i * 0.38
            ax.text(cx + 0.15, y, item, ha="left", va="center",
                    fontsize=8.5, color="#c9d1d9",
                    fontfamily="monospace")
            ax.plot(cx + 0.05, y, "o", color=color, markersize=4)

    # Arrows between columns
    arrow_style = dict(arrowstyle="->,head_width=0.25,head_length=0.12",
                       color="#58a6ff", lw=1.8)
    ax.annotate("", xy=(3.6, 2.2), xytext=(2.8, 2.2), arrowprops=arrow_style)
    ax.annotate("", xy=(7.6, 2.2), xytext=(6.8, 2.2), arrowprops=arrow_style)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "feature_pipeline.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Generated feature_pipeline.png")


if __name__ == "__main__":
    generate_architecture_diagram()
    generate_performance_summary()
    generate_lr_schedule()
    generate_feature_pipeline()
    print("\nAll graphs generated.")
