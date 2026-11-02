# config.py

#####################################################
#      This option is for the main experiment.      #
#####################################################

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

DATASET_NAME = "svamp"
DATASET_CONFIG = "main"

TRAIN_SIZE = 80
TEST_SIZE = 100

MAX_NEW_TOKENS = 160
TEMPERATURE = 0.0
BATCH_SIZE = 8

NUM_ITERATIONS = 2
NUM_CANDIDATES_PER_ITER = 2

LAMBDA_FORMAT = 0.2
LAMBDA_TOKEN = 0.1
TOKEN_BASELINE = 300

RESULT_DIR = "/root/final_project/results/runs"
SEED_PROMPT_PATH = "/root/final_project/prompts/seed_prompts.json"

SCORE_MODE = "accuracy_only"
RUN_NAME = "svamp_accuracy_only"



#####################################################
#    This option is for recoding the demo video.    #
#####################################################

"""MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

DATASET_NAME = "svamp"
DATASET_CONFIG = "main"

TRAIN_SIZE = 16
TEST_SIZE = 16

MAX_NEW_TOKENS = 120

TEMPERATURE = 0.0

BATCH_SIZE = 8

NUM_ITERATIONS = 1
NUM_CANDIDATES_PER_ITER = 2

SCORE_MODE = "multi_objective"

RUN_NAME = "demo_svamp_multi_objective"

LAMBDA_FORMAT = 0.2
LAMBDA_TOKEN = 0.1
TOKEN_BASELINE = 300

RESULT_DIR = "/root/final_project/results/runs"
SEED_PROMPT_PATH = "/root/final_project/prompts/seed_prompts.json\""""

