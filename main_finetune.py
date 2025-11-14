import numpy as np
from itertools import product
from multiprocessing import Pool
import main_pipeline as mp
import hyperparams as hp

def run_with_hyperparams(params_tuple):
    """Run main pipeline with specific hyperparameters"""
    lr, decay, warmup = params_tuple
    
    hp.LEARNING_RATE = lr
    hp.WEIGHT_DECAY = decay
    hp.WARMUP = warmup
        
    return mp.main('training_data.csv', 'regressor', DEBUG_MODE=False, GRAPH_MODE=False)

def finetune():
    lrs = np.arange(0.000005, 0.000011, 0.000001)
    decays = np.arange(0.003, 0.011, 0.001)
    warmups = np.arange(7000, 13000, 1000)

    all_hyperparams = list(product(lrs, decays, warmups))
        
    with Pool(processes=min(2, len(all_hyperparams))) as pool:
        pool.map(run_with_hyperparams, all_hyperparams)
    
if __name__ == "__main__":
    finetune()

