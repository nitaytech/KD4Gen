# typing imports
import os
from typing import List, Tuple, Dict, Any, Union, Callable, Optional, Collection
from pathlib import Path
from pandas import DataFrame
from numpy import ndarray


DATA_DIR = 'data'
EXPERIMENTS_DIR = 'experiments'

# model
STUDENT_MODEL_NAME = 't5-small'
TEACHER_MODEL_NAME = 't5-large'
LOSS_IGNORE_ID = -100
DECODER = 'decoder-only'
ENC_DEC = 'encoder-decoder'
LANGUAGE_MODEL = 'language-model'

# data
SPLIT_TRAIN = 'train'
SPLIT_VAL = 'dev'
SPLIT_VAL_PPL = 'dev_ppl'
SPLIT_TEST = 'test'
SPLIT_UNLABELED = 'unlabeled'
SPLIT_FT = 'ft'
TEXT_COL = 'text'
TARGET_COL = 'target'
ID_COL = 'id'
GEN_COL = 'generated'
TASK_PROMPT = ''
TOKENS_COL = 'tokens'
LOGITS_COL = 'logits'

# trainer
BATCH_SIZE = 32
MAX_GPU_BATCH_SIZE = 8
PRECISION = 'no'  # 'fp16'
NUM_EPOCHS = 20
WEIGHT_DECAY = 1e-5
LEARNING_RATE = 1e-5
OPTIMIZER_EPS = 1e-8
WARMUP_STEPS = 100
METRIC_FOR_EVAL = 'rouge2_f'

# generation
MAX_INPUT_LENGTH = 512
MAX_LABELS_LENGTH = 128
MAX_LENGTH = MAX_INPUT_LENGTH + MAX_LABELS_LENGTH
NUM_BEAMS = 4

# evaluation
BERTSCORE_MODEL_NAME = "microsoft/deberta-base-mnli"

# experiments
SEED = 42

# kd
KD_LOSS_TYPE = 'logits'
SAMPLIMG_STEPS = 100