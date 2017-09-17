#############################################################################################################
### SETTINGS : Hyper-parameters, file paths, model settings etc. ###

import re

DATA_PATH = "./data/cornell_movie-dialogs_corpus/movie_{}.txt" # Where is the data at?
SPLIT = " +++$+++ " # Split char in the data
VOCAB_PATH = "./data/vocab.{}"

DIGIT_RE = re.compile("\d")
WORD_SPLIT_RE = re.compile("([.,!?\"-<>:;)(])")
NORMALIZE_DIGITS = True
MAX_VOCAB_SIZE = 1e5
TRAIN_PCT = 0.9

BUCKETS = [(8, 10), (12, 14), (16, 19)]

# Special char ids
PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

THRESHOLD = 2 # Minimum number of occurences for vocab list


NUM_SAMPLES = 512
HIDDEN_SIZE = 356
NUM_LAYERS = 3

BATCH_SIZE = 64
LEARNING_RATE = 0.5
MAX_GRADIENT_NORM = 5.0

CKPT_PATH = 'checkpoints'
SUMMARY_PATH = 'logs'
GENERATED_PATH = 'generated/generated-user-bot-coversation-{}'

SAVE_EVERY = 500
PRINT_EVERY = 100
EVAL_EVERY = 500
MAX_ITER = 2500000

# Dynamically updated based on dataset with file i/o
ENC_VOCAB_LEN = 42996 # 43459
DEC_VOCAB_LEN = 43097 # 43580 
