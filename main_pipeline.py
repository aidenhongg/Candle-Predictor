import os

import pandas as pd

from main_preprocess import preprocess
from pipeline import WindowLoader, train
from graphing import graph_masks, graph_deltas, plot_loss


def main(filename='training_data.csv', task='classifier', DEBUG_MODE=False, GRAPH_MODE=True):
    processed_file = filename.replace('.csv', '_p.csv')
    if processed_file not in os.listdir(os.getcwd()):
        preprocess(filename)

    df = pd.read_csv(processed_file)

    # Prepare graphing data before splitting (avoids holding full df in memory)
    if task == 'classifier':
        df_graph = df['close'].iloc[-5000::5].reset_index(drop=True)
        labels = df['is_mask'].iloc[-5000::5].reset_index(drop=True)
    elif task == "regressor":
        df_masked = df[df['is_mask'] == 1]
        labels = df_masked['close_diff'].iloc[-5000::5].reset_index(drop=True)

    split = int(len(df) * 0.9)
    train_data = WindowLoader(df[:split], task)
    test_data = WindowLoader(df[split:], task)
    del df

    loss_record, predictions = train(train_data, test_data, task, debug_mode=DEBUG_MODE)

    if GRAPH_MODE:
        predictions = pd.Series(predictions.squeeze().cpu().numpy()[:, -1])[-5000::5].reset_index(drop=True)
        if task == 'classifier':
            graph_masks(df_graph, predictions, labels)
        elif task == 'regressor':
            graph_deltas(predictions, labels)

        plot_loss(loss_record, task)


if __name__ == "__main__":
    main()
