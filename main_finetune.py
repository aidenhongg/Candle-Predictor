import random
import multiprocessing as mp
from itertools import product

import numpy as np

NUM_PROCESSES = 10


def run_with_hyperparams(args):
    """Train a single model with the given hyperparameter combination on a specific GPU."""
    import os
    gpu_id, (lr, decay, warmup) = args

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import torch
    import main_pipeline
    import hyperparams as hp

    torch.cuda.set_device(0)

    hp.LEARNING_RATE = lr
    hp.WEIGHT_DECAY = decay
    hp.WARMUP = warmup

    return main_pipeline.main('training_data.csv', 'regressor', DEBUG_MODE=False, GRAPH_MODE=False)


def run_seed_regressor(args):
    """Train a regressor with a specific random seed on a specific GPU."""
    import os
    gpu_id, seed = args

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import torch
    import main_pipeline
    import hyperparams as hp

    torch.cuda.set_device(0)

    hp.SEED = seed
    return main_pipeline.main('training_data.csv', 'regressor', DEBUG_MODE=False, GRAPH_MODE=False)


def run_seed_classifier(args):
    """Train a classifier with a specific random seed on a specific GPU."""
    import os
    gpu_id, seed = args

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import torch
    import main_pipeline
    import hyperparams as hp

    torch.cuda.set_device(0)

    hp.SEED = seed
    return main_pipeline.main('training_data.csv', 'classifier', DEBUG_MODE=False, GRAPH_MODE=False)


def grid_search():
    """Run a grid search over learning rate, weight decay, and warmup steps."""
    learning_rates = np.arange(0.000005, 0.000011, 0.000001)
    weight_decays = np.arange(0.003, 0.011, 0.001)
    warmup_steps = np.arange(7000, 13000, 1000)

    all_combinations = list(product(learning_rates, weight_decays, warmup_steps))

    tasks = [(i % NUM_PROCESSES, params) for i, params in enumerate(all_combinations)]

    with mp.Pool(processes=min(NUM_PROCESSES, len(all_combinations))) as pool:
        pool.map(run_with_hyperparams, tasks)


def seed_search():
    """Train both regressor and classifier across multiple random seeds."""
    regressor_seeds = [random.randint(0, 2**32 - 1) for _ in range(10)]
    tasks = [(i % NUM_PROCESSES, seed) for i, seed in enumerate(regressor_seeds)]

    with mp.Pool(processes=min(NUM_PROCESSES, len(regressor_seeds))) as pool:
        pool.map(run_seed_regressor, tasks)

    classifier_seeds = [random.randint(0, 2**32 - 1) for _ in range(10)]
    tasks = [(i % NUM_PROCESSES, seed) for i, seed in enumerate(classifier_seeds)]

    with mp.Pool(processes=min(NUM_PROCESSES, len(classifier_seeds))) as pool:
        pool.map(run_seed_classifier, tasks)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    seed_search()
