import os
import pandas as pd
from main_preprocess import preprocess
from pipeline import WindowLoader, train 
from graphing import *

def main(FILENAME = 'training_data.csv', task = 'classifier', DEBUG_MODE = False, GRAPH_MODE = True):
    processed_file = FILENAME.replace('.csv', '_p.csv')
    if not processed_file in os.listdir(os.getcwd()):
        preprocess('training_data.csv')

    df = pd.read_csv(processed_file)

    # for graphing - do beforehand to delete df
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

    loss_record, predictions = train(train_data, test_data, task, debug_mode = DEBUG_MODE)

    if GRAPH_MODE:
        predictions = pd.Series(predictions.squeeze().cpu().numpy())[-5000::5].reset_index(drop=True)
        if task == 'classifier':
            graph_masks(df_graph, predictions, labels) 
        elif task == 'regressor':
            graph_deltas(predictions, labels)
        
        plot_loss(loss_record, task)        

if __name__ == "__main__":
    main(task='classifier')
    # main(task='regressor')
    # print('classifier done, press enter to proceed')
    input()
