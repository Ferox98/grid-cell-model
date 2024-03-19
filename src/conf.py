# Define Grid parameters
GRID_ROWS = 6
GRID_COLS = 6
NUM_HIDDEN = 1
NUM_SAMPLES = 108
# Define Network parameters
INPUT_DIM = 4
HIDDEN_LAYERS = 1
HIDDEN_SIZE = 10
OUTPUT_DIM = GRID_ROWS * GRID_COLS

REPRESENTATION_SPACE = 4
# Define training parameters
NUM_EPOCHS = 8
LEARNING_RATE = 0.001
GAMMA = 0.995
SEQ_LEN = 36
NUM_BATCHES = 1
BATCHES_PER_EPOCH = 4
# How many iterations of the first epoch should the agent be allowed to explore?
EXPLORE_ITERATIONS = 25
# How many environments to sequentially train TEM on
NUM_ENVIRONMENTS  = 10
# L1 Regularization Parameter
LAMBDA = 10.0
PRINT_ITER = False
# Hopfield Net params
# Inverse temperatures to prevent meta-stable states between local minima from forming
BETA = 5.0
# To improve performance as training progresses, old memories have to be replaced. 
FORGET_ITER = 3
FORGET_PCT = 0.2