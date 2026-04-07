import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import hyperparams as hp


def graph_masks(points, pred_mask, label_mask):
    """Plot predicted vs actual trend classification overlaid on price data."""

    def graph_trends(label_type, mask):
        plt.figure(figsize=(int(len(points) / 15), 10))
        close_values = points.values
        for i in range(len(points) - 1):
            color = 'red' if mask[i] == 1 else 'blue'
            plt.plot([i * 5, (i + 1) * 5], [close_values[i], close_values[i + 1]],
                     color=color, linewidth=2)

        legend_elements = [
            Line2D([0], [0], color='red', linewidth=4, label='Trending'),
            Line2D([0], [0], color='blue', linewidth=4, label='Non-trending'),
        ]
        plt.legend(handles=legend_elements)
        plt.xlabel('Time (minutes)')
        plt.ylabel('Close Price')
        plt.title('Price Trend Classification ' + ('(Predicted)' if label_type == 'pred' else '(Actual)'))
        plt.savefig(f'./graphing/graphs/{hp.SEED}{label_type}_classes.png', dpi=400)
        plt.close()

    graph_trends('pred', pred_mask)
    graph_trends('label', label_mask)


def graph_deltas(predictions, labels):
    """Plot predicted vs actual close price deltas."""
    plt.figure(figsize=(int(len(predictions) / 15), 10))

    x_values = [i * 5 for i in range(len(predictions))]

    plt.plot(x_values, labels.values, color='black', linewidth=4, label='Actual')
    plt.plot(x_values, predictions.values, color='green', linewidth=4, label='Predicted')

    plt.xlabel('Time (minutes)')
    plt.ylabel('Close Delta')
    plt.title('Close Delta Predictions vs Actual')
    plt.legend()

    plt.savefig(f'./graphing/graphs/{hp.SEED}delta_comparison.png', dpi=400)
    plt.close()


def plot_loss(loss_record, task):
    """Plot training loss convergence curve."""
    loss_values = [loss.cpu().item() for loss in loss_record]
    sampled_loss = loss_values[::5]

    width = len(sampled_loss) / 20
    plt.figure(figsize=(width * 5, 10))

    plt.plot(sampled_loss, color='black', linewidth=2)

    plt.title('Training Loss Convergence')
    plt.xlabel('Training Steps (Per 2500 Batches)')
    plt.ylabel('Loss')

    plt.savefig(f'./graphing/graphs/{hp.SEED}{task}_convergence.png', dpi=400)
    plt.close()


if __name__ == "__main__":
    # Smoke test: generate a sample loss plot
    test_loss_record = [torch.tensor(5.0 - i * 0.02 + np.random.rand() * 0.5) for i in range(200)]
    plot_loss(test_loss_record, 'test')
    print("Test loss plot generated successfully")
