from yacs.config import CfgNode as CN

###########################
# Config definition
###########################
_C = CN()
_C.OUT_DIR = "./output"
_C.SEED = 42
_C.USE_CUDA = True
_C.VERBOSE = True
_C.DROP_LAST = False
_C.DROP_OUT = 0.0
_C.LOG_STEP = 5

###########################
# Base
###########################
_C.EPOCHS = 100
_C.LEARNING_RATE = 1e-3
_C.BATCH_SIZE = 16
_C.WEIGHT_DECAY = 5e-4
_C.MOMENTUM = 0.9

###########################
# Model
###########################
_C.MODEL = CN()
_C.MODEL.NAME = ""
_C.BACKBONE = ""

###########################
# Optimizer
###########################
_C.OPTIM = CN()
_C.OPTIM.NAME = ""

###########################
# Transforms
###########################
_C.TRANSFORM = CN()
_C.TRANSFORM.NAME = []
_C.TRANSFORM.AUGPROB = 0.5

# ColorJitter (brightness, contrast, saturation, hue)
_C.TRANSFORM.COLORJITTER_B = 1
_C.TRANSFORM.COLORJITTER_C = 1
_C.TRANSFORM.COLORJITTER_S = 1
_C.TRANSFORM.COLORJITTER_H = 0.05

###########################
# Dataset
###########################
_C.DATASET = CN()
_C.DATASET.ROOT = ""
_C.DATASET.NUM_CLASSES = 0
_C.DATASET.SOURCE_DOMAINS = ()
_C.DATASET.TARGET_DOMAINS = ()

###########################
# GDRNet
###########################
_C.GDRNET = CN()
_C.GDRNET.BETA = 0.5
_C.GDRNET.TEMPERATURE = 0.1
_C.GDRNET.SCALING_FACTOR = 4.

###########################
# Fishr
###########################
_C.FISHR = CN()
_C.FISHR.NUM_GROUPS = 0
_C.FISHR.EMA = 0.
_C.FISHR.PENALTY_ANNEAL_ITERS = 0
_C.FISHR.LAMBDA = 0.

###########################
# DRGen
###########################
_C.DRGEN = CN()
_C.DRGEN.N_CONVERGENCE = 0
_C.DRGEN.N_TOLERANCE = 0
_C.DRGEN.TOLERANCE_RATIO = 0.
