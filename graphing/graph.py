from matplotlib.lines import Line2D
import hyperparams as hp # please take out later
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

def graph_masks(points, pred_mask, label_mask):
    
    def graph_trends(type, mask):
        plt.figure(figsize=(int(len(points) / 15), 10))
        close_values = points.values
        for i in range(len(points)):
            color = 'red' if mask[i] == 1 else 'blue'
            if i < len(points) - 1:
                plt.plot([i*5, (i+1)*5], [close_values[i], close_values[i+1]], 
                        color=color, linewidth=2)

        # Create legend
        legend_elements = [Line2D([0], [0], color='red', linewidth=4, label='red - trending'),
                          Line2D([0], [0], color='blue', linewidth=4, label='blue - non trending')]
        plt.legend(handles=legend_elements)

        plt.xlabel('Time (minutes)')
        plt.ylabel('Close Price')
        plt.title('Price Trend Classification ' + ('(Predicted)' if type == 'pred' else '(Actual)'))
        plt.savefig(f'./graphing/graphs/{hp.SEED}{type}_classes.png', dpi=400)
        plt.close()
    
    graph_trends('pred', pred_mask)
    graph_trends('label', label_mask)

def graph_deltas(predictions, labels):
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


def plot_loss(loss_record : list, task : str):
    loss_record = [l.cpu().item() for l in loss_record]
    sampled_loss = loss_record[::5]
    
    width = len(sampled_loss) / 20
    plt.figure(figsize=(width * 5, 10))
    
    plt.plot(sampled_loss, color='black', linewidth=2)
    
    plt.title('Training Loss Convergence')
    plt.xlabel('Training Steps (Per 2500 Batches)')
    plt.ylabel('Loss')
    
    plt.savefig(f'./graphing/graphs//{hp.SEED}{task}_convergence.png', dpi=400)
    plt.close()

def test_plot_loss():
        
    loss_record = [torch.tensor(5.0 - i * 0.02 + np.random.rand() * 0.5) for i in range(200)]
    
    plot_loss(loss_record)
    print("Test loss plot generated successfully")

def test_graph_masks():
        length = 100
        points = pd.Series(np.random.randn(length).cumsum() + 100)
        pred_mask = pd.Series(np.random.randint(0, 2, length))
        label_mask = pd.Series(np.random.randint(0, 2, length))
        
        graph_masks(points, pred_mask, label_mask)
        print("Test graphs generated successfully")

if __name__ == "__main__":
    test_plot_loss()
    test_graph_masks()