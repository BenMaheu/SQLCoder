from dataclasses import dataclass

NEW_TOKENS = {"{": "[LBRACE]", "}": "[RBRACE]", "[": "[LBRACK]", "]": "[RBRACK]"}
OP2IDX = {"=": 0, ">": 1, "<": 2}
IDX2OP = {0: "=", 1: ">", 2: "<"}
AGGS_STR_ORDERED = ["NULL", "MAX", "MIN", "COUNT", "SUM", "AVG"]
AGGS_ORDERED = [None, "MAX", "MIN", "COUNT", "SUM", "AVG"]
SPECIAL_TOKENS = {'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>'}

DATA_PATH = r"../data/dataset_files"
SQL_DB_PATH = r"../data/db_files"

# -------------- Processor PARAMS ----------------
ALLOWED_MODES = ["structured_output", "human_readable_output", "runnable_output"]

# -------------- TOKENIZER PARAMS ------------------
COL_SEP = " | "  # Column separator in the input prompt
SAMPLE_SEP = " ; "  # Samples separator in the input prompt
MAX_INPUT_LENGTH = 128
MAX_OUTPUT_LENGTH = 128

# ------------- TRAINING DEFAULT CONFIG ---------------
BATCH_SIZE = 16
EPOCHS = 5
GENERATION_MAX_LENGTH = 128


@dataclass
class Config:
    model_name: str = "google/flan-t5-small"
