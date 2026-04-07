# Data
WINDOW_SIZE = 480
BATCH_SIZE = 64

# Optimizer
LEARNING_RATE = 0.000006
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 0.05

# Scheduler (cosine annealing with warm restarts)
WARMUP = 9000
T0 = 10000
T_MULT = 2

# Model
DROPOUT = 0.1

# Early stopping
STOP_BUFFER = 0.00001
PATIENCE = 5

# Preprocessing (EWMA smoothing)
VEL_ALPHA = 0.3
ACCEL_ALPHA = 0.3

# Reproducibility
SEED = None