import numpy as np
from itertools import product
import multiprocessing as mp
import random


NUM_PROCESSES = 10

def run_with_hyperparams(args):
    import os
    gpu_id, params_tuple = args
    lr, decay, warmup = params_tuple
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import main_pipeline as pipeline
    import hyperparams as hp
    import torch
    torch.cuda.set_device(0)  # After setting CUDA_VISIBLE_DEVICES, use device 0
    
    hp.LEARNING_RATE = lr
    hp.WEIGHT_DECAY = decay
    hp.WARMUP = warmup
        
    return pipeline.main('training_data.csv', 'regressor', DEBUG_MODE=False, GRAPH_MODE=False)

def run_seeds_reg(args):
    import os

    gpu_id, seed = args
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import main_pipeline as pipeline
    import hyperparams as hp
    import torch
    torch.cuda.set_device(0)
    
    hp.SEED = seed
    return pipeline.main('training_data.csv', 'regressor', DEBUG_MODE=False, GRAPH_MODE=False)

def run_seeds_class(args):
    import os
    gpu_id, seed = args
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import main_pipeline as pipeline
    import hyperparams as hp
    import torch
    torch.cuda.set_device(0)
    
    hp.SEED = seed
    return pipeline.main('training_data.csv', 'classifier', DEBUG_MODE=False, GRAPH_MODE=False)


def finetune_params():
    lrs = np.arange(0.000005, 0.000011, 0.000001)
    decays = np.arange(0.003, 0.011, 0.001)
    warmups = np.arange(7000, 13000, 1000)

    all_hyperparams = list(product(lrs, decays, warmups))
    
    # Assign GPU IDs cyclically
    tasks = [(i % NUM_PROCESSES, params) for i, params in enumerate(all_hyperparams)]
        
    with mp.Pool(processes=min(NUM_PROCESSES, len(all_hyperparams))) as pool:
        pool.map(run_with_hyperparams, tasks)

def seed_search():
    seeds = [random.randint(0, 2**32 - 1) for _ in range(10)]

    tasks = [(i % NUM_PROCESSES, seed) for i, seed in enumerate(seeds)]
    
    with mp.Pool(processes=min(NUM_PROCESSES, len(seeds))) as pool:
        pool.map(run_seeds_reg, tasks)

    seeds = [random.randint(0, 2**32 - 1) for _ in range(10)]
    tasks = [(i % NUM_PROCESSES, seed) for i, seed in enumerate(seeds)]

    with mp.Pool(processes=min(NUM_PROCESSES, len(seeds))) as pool:
        pool.map(run_seeds_class, tasks)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    seed_search()
