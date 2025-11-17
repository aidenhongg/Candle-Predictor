import numpy as np
from itertools import product
from multiprocessing import Pool
import main_pipeline as mp
import hyperparams as hp
import torch

NUM_PROCESSES = 5

def run_with_hyperparams(params_tuple):
    lr, decay, warmup = params_tuple
    
    hp.LEARNING_RATE = lr
    hp.WEIGHT_DECAY = decay
    hp.WARMUP = warmup
        
    return mp.main('training_data.csv', 'regressor', DEBUG_MODE=False, GRAPH_MODE=False)

def run_seeds_reg(seed):
    hp.SEED = seed

    return mp.main('training_data.csv', 'regressor', DEBUG_MODE=False, GRAPH_MODE=False)

def run_seeds_class(seed):
    hp.SEED = seed

    return mp.main('training_data.csv', 'classifier', DEBUG_MODE=False, GRAPH_MODE=False)


def finetune_params():
    lrs = np.arange(0.000005, 0.000011, 0.000001)
    decays = np.arange(0.003, 0.011, 0.001)
    warmups = np.arange(7000, 13000, 1000)

    all_hyperparams = list(product(lrs, decays, warmups))
        
    with Pool(processes=min(NUM_PROCESSES, len(all_hyperparams))) as pool:
        pool.map(run_with_hyperparams, all_hyperparams)

def seed_search():
    seeds = [torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(10)]
    
    with mp.Pool(processes=min(NUM_PROCESSES, len(seeds))) as pool:
        pool.map(run_seeds_reg, seeds)

    seeds = [torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(10)]

    with mp.Pool(processes=min(NUM_PROCESSES, len(seeds))) as pool:
        pool.map(run_seeds_class, seeds)


if __name__ == "__main__":
    seed_search()

