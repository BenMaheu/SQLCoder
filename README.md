# Install dependencies
Install all dependencies using poetry in the `/src` directory:
```
poetry install
```

Poetry does not have a correct torch dependency specification yet, so please install it manually.
```
pip install torch
```

For accelerate : 
```
pip install -U "transformers[torch]"
```

# Run Streamlit app
To run the Streamlit app, use the following command:
```
streamlit run src/app.py
```
![](src/assets/streamlit_app.gif)

# Run unit tests
To run unit tests, use the following command from the `/src` directory:
```
python -m unittest discover
```

# Project architecture
```
.
├── README.md                 # Take-home assignment.
├── data
│   ├── dataset_files/        # Preprocessed WikiSQL dataset (train/val/test .jsonl)
│   ├── db_files/             # SQLite database files for query execution
│   └── test_schemas.pkl      # Pickled schemas for Streamlit demo (UI mode)
└── src
    ├── README.md             # Developer-focused documentation for src/ codebase
    ├── app.py                # Streamlit web application for interactive Text2SQL inference
    ├── assets/
    │   ├── data_distrib.png  # Data distribution plot used in the README report
    │   └── streamlit_app.gif # Animated demo of the Streamlit app
    ├── config.py             # Global constants: aggregation ops, operator mappings, etc.
    ├── data/
    │   ├── eda.py            # Exploratory data analysis utilities (plotting distributions)
    │   ├── loader.py         # Dataset loader (train/val/test split management)
    │   ├── process.py        # Data preprocessing logic (prompt creation, normalization)
    │   └── wiki_sql_loader.py# WikiSQL-specific dataset parser and transformer
    ├── evaluation/
    │   └── metrics.py        # Evaluation metrics
    ├── extract_schemas.py    # Extracts DB schema metadata for each table (headers, types), for Streamlit app
    ├── inference.py          # Inference example script using trained models
    ├── model/
    │   ├── text_to_sql.py    # Core seq2seq model class (Flan-T5 fine-tuning and inference)
    │   └── utils.py          # Model utilities (load_model_and_tokenizer, load_model_from_checkpoint)
    ├── models/               # Directory containing trained checkpoints and saved Hugging Face models
    ├── poetry.lock           # Dependency lock file for Poetry
    ├── pyproject.toml        # Poetry configuration (dependencies and project metadata)
    ├── sanitizer/
    │   ├── sql_sanitizer.py  # SQLSanitizer module
    ├── SQLCoder.ipynb        # Jupyter notebook with whole pipeline from data loading to evaluation
    ├── tests/
    │   ├── test_processor.py # Unit tests for data preprocessing
    │   ├── test_sql_sanitizer.py # Unit tests for SQLSanitizer logic
    │   └── test_utils.py     # Tests for shared utility functions
    ├── train.py              # Main training script for model fine-tuning
    ├── train_base_hro_model  # Example command line script to train Flan-T5 base (human-readable output)
    ├── train_small_hro_model # Example command line script to train Flan-T5 small (human-readable output)
    └── utils.py              # General helper functions (text normalization, token stats, etc.)

```


# Project report

## Introduction
Based on the WikiSQL dataset provided, the objective is to train a model capable of converting natural language queries into SQL queries. 
The dataset includes `human_readable` SQL queries that are logically correct but not immediately functional, requiring adjustments such as proper table names and value formatting (backticks, quotation marks...).

The dataset also includes `types` that can help us to either include them in the input prompt or to post-process the 
output query (`real` values should be without any backtick or quotation mark, `text` values should be quoted).

To tackle this problem, I thought of 3 different methods using the same base architecture (FLAN-T5):
* Fine-tuning a seq2seq model to generate structured output that can be further used to reconstruct functional SQL queries. (`structured_output`)
* Fine-tuning a seq2seq model to generate human readable output as they are included in the WikiSQL dataset then use a SQLSanitizer class to post-process the output queries to make them functional. (`human_readable_output`)
* Fine-tuning a seq2seq model to generate runnable SQL queries directly. (`runnable_output`)

For each of the 3 methods, we have trained a FLAN-T5 model using the HuggingFace transformers library.


## Base Architecture

All experiments were conducted using FLAN-T5 (small and base variants) from Hugging Face’s transformers library.
Each model was fine-tuned for 5 epochs with:
* `batch_size = 16`
* `learning_rate = 5e-5`
* Linear scheduler with`warmup_steps = 0`
* Optimizer: `AdamW`
* Trainer: `Seq2SeqTrainer`

Training ran on a single NVIDIA T4 GPU (Google Colab).

---

## Data
We trained, validated and evaluated on the entire WikiSQL dataset provided in the `data/dataset_files/` folder:
```
Train size: 49190 | Val size: 8421 | Test size: 15878
```

Operator and aggregator mappings:
```
OP2IDX = {"=": 0, ">": 1, "<": 2}
IDX2OP = {0: "=", 1: ">", 2: "<"}
AGGS_STR_ORDERED = ["NULL", "MAX", "MIN", "COUNT", "SUM", "AVG"]
```
Each observation in the dataset contains a `sql` dictionary with the following structure:
```python
{'human_readable': 'SELECT Notes FROM table WHERE Current slogan = SOUTH AUSTRALIA',
 'sel': 5,
 'agg': 0,
 'conds': {'col': [3], 'op': [0], 'val': ['SOUTH AUSTRALIA']}}
```
Where `sel` is the index of the selected column, `agg` is the index of the aggregate function in `AGGS_STR_ORDERED` 
and `conds` contains lists of indices for columns, operators and values for each condition in the WHERE clause.

The `op` indices correspond to the `OP2IDX` mapping.

A short Exploratory Data Analysis on the distribution of the number of columns and rows per SQL table, the types of 
each column, the number of operators, aggregate functions and number of conditions per SQL query helps us better understand our data.
![](src/assets/data_distrib.png)

---

## Methods

### Human readable output with post-processing (SQL Sanitizing)
In this method the model is trained to generate human readable SQL queries from a well-tailored prompt that should describe the SQL table queried.
To provide the model with sufficient information on the column names, we format the input prompt as follows:

**Prompt template :**
```
task: text-to-sql
question: Tell me what the notes are for South Australia 
columns: State/territory | Text/background colour | Format | Current slogan | Current series | Notes
```

**Output target**:
```
SELECT Notes FROM table WHERE Current slogan = SOUTH AUSTRALIA
```

Note that the table name is not included in the prompt since it is a placeholder in the human readable SQL queries. 
This placeholder is further replaced once passed to the SQLSanitizer class that builds a runnable SQL query from the raw generated output.

One should also note that the values in the output target are not quoted, thus leading to no need to include types in the input prompt as the quotation process will be handled by the SQLSanitizer.

Once the model is trained, we can use the SQLSanitizer class to post-process the generated SQL queries to make them functional.
Example of usage:
```python
from sql_sanitizer import SQLSanitizer

sanitized_query = sanitizer.sanitize(pred_sql, table)
```

Where `pred_sql` is the raw generated SQL query from the model and `table` is the table metadata included in the dataset sample.

Using this method the distribution of length of tokens is as follows:
```
	INPUT/OUTPUT TOKEN DISTRIBUTION
Input Mean: 50.82, %-Input > 256: 0.00%,  %-Input > 128: 0.23%, %-Input > 64: 11.69%
Output Mean: 21.47, %-Output > 256: 0.00%, %-Output > 128: 0.00%, %-Output > 64: 0.03%
```
Thus, ensuring that a model with a maximum input/output length of `128 tokens` should be sufficient for most part of the examples for this task.


During our experiments we tried to replace `"|"` separator in the columns list with a special token column separator 
that was added to the tokenizer but it led to worse results as the model struggled to identify column names correctly.

---

### Structured output with post-processing
In this method, the model is trained to generate a structured output from the same input prompt as the previous method.

In a first experiment, we tried using the original `sql` dictionary structure from the WikiSQL dataset as output target. 
However selected columns and aggregate functions were represented using integers which led to poor results since the 
model struggled to generate these integers correctly or to find the semantic relationship between the generated integer and the column indices. 

This is why we decided to format the output target using the actual column names and aggregate functions as strings.

Thus the output target is formatted as follows:
```
[LBRACE]"sel": "Notes", "agg": "NULL", "conds": [LBRACE]"col": [LBRACK]"Current slogan"[RBRACK], "op": [LBRACK]"="[RBRACK], "val": [LBRACK]"SOUTH AUSTRALIA"[RBRACK][RBRACE][RBRACE]
```
Where `LBRACE`, `RBRACE`, `LBRACK` and `RBRACK` are special tokens added to the tokenizer vocabulary to represent respectively `{`, `}`, `[`, and `]`.
We added these special tokens as the original FLAN-T5 tokenizer does not include them by default.
```python
enc = tokenizer(["{'a': 5}"])
dec = tokenizer.batch_decode(enc["input_ids"])[0]
print(dec)
# <unk> 'a': 5<unk></s>
```
>If we were to continue working on the project, an interesting lead would be to use a tokenizer trained on code data such as the `CodeT5` tokenizer that should include these characters in the vocabulary.

From there reconstructing the functional SQL query is straight-forward if we make sure during this step to pass column 
types as argument in order to format values correctly (quotation marks for text values, no quotes for real values...).

I encountered some minor issues when reconstructing the SQL query as some `"real"` columns contain numbers with `","`as thousands separator (e.g. `1,234`).
However a simple regex can be used to detect these cases and remove the commas before converting the value to a float.

Using this method the distribution of length of tokens is as follows:
```
	INPUT/OUTPUT TOKEN DISTRIBUTION
Input Mean: 50.82, %-Input > 256: 0.00%,  %-Input > 128: 0.23%, %-Input > 64: 11.69%
Output Mean: 73.19, %-Output > 256: 0.00%, %-Output > 128: 0.04%, %-Output > 64: 89.71%
```
Thus, ensuring that a model with a maximum input/output length of `128 tokens` should be sufficient for most part of the examples for this task.

---

### Runnable output
In this method, the model is trained to generate runnable SQL queries directly from a slightly different input prompt than the previous methods.

As we want to generate directly executable SQL queries, we need to include the types of each column in the input prompt to help the model understand how to format values correctly.
**Prompt template :**
```
task: text-to-sql
question: Tell me what the notes are for South Australia 
columns: State/territory | Text/background colour | Format | Current slogan | Current series | Notes
types: text | text | text | text | text | text
```

**Output target**:
```
SELECT [backtick]Notes[backtick] FROM <table> WHERE [backtick]Current slogan[backtick] = "SOUTH AUSTRALIA"
```
We used `[backtick]` special token for the same reasons as previously mentioned regarding `{`, `}`, `[`, and `]`.
Here a `<table>` token was added to the tokenizer vocabulary as well for easier post-processing and to ensure the model learns to generate it correctly.

Using this method the distribution of length of tokens is as follows:
```
	INPUT/OUTPUT TOKEN DISTRIBUTION
Input Mean: 64.61, %-Input > 256: 0.02%,  %-Input > 128: 0.69%, %-Input > 64: 41.26%
Output Mean: 49.84, %-Output > 256: 0.00%, %-Output > 128: 0.01%, %-Output > 64: 12.43%
```
Thus, ensuring that a model with a maximum input/output length of `128 tokens` should be sufficient for most part of the examples for this task.

---

### SQLSanitizer
The SQLSanitizer module ensures that raw SQL queries generated by the model are syntactically valid and executable against the WikiSQL databases.
It performs post-processing operations such as quoting column names, correcting value formatting and resolving column match.
For instance:
```python
from sql_sanitizer import SQLSanitizer
sanitizer = SQLSanitizer()
pred_sql = "SELECT Note FROM table WHERE Current slogan = SOUTH AUSTRALIA"
sanitized_query = sanitizer.sanitize(pred_sql, table)

# Output: SELECT `Notes` FROM table_2_12888640_1 WHERE `Current slogan` = "SOUTH AUSTRALIA";
```

#### a. Checks if `SELECT` is the first statement for safety
The `sanitize()` method first checks if the generated query starts with a `SELECT` clause.

#### b. Cleanup and Normalization
* Removes custom special tokens (e.g. `[LBRACE]`, `[RBRACK]`) using utility functions `unformat_string()` and `strip_specials()`.
* Trims trailing characters and enforces the query to start with a SELECT clause for safety.
* Strips semicolons and re-adds them once the final query is clean.

#### c. Column name quoting and string similarity
The sanitizer uses `difflib.get_close_matches` to perform string similarity based on the Ratcliff-Obershelp algorithm.
This algorithm is based on a score that computes the longest common subsequence between two strings.
> One interesting lead for future work would be to try other string similarity algorithms such as Levenshtein distance (`fuzzywuzzy` library) 
> which would count insertions, substitions and deletions.

That way, the sanitizer allows tolerance for typos or slight naming differences between generated column names and
actual table columns. (Implementing this led to a significant win in performance during evaluation)

This step is handled by:
* `best_match_column()`: finds the best matching column name from the table for a given generated column name.
* `quote_select_column()`: ensures aggregates like COUNT, SUM, AVG are reconstructed correctly.

#### d. WHERE Clause Sanitization
The `sanitize_where_conditions()` method detects and formats condition triplets (column, operator, value) :
* Recognizes valid operators (`=`, `<`, `>`)
* Quotes string literals (unless numeric)
* Preserves logical connectors (`AND`, `OR`)

> This method could also use types from the table to ensure values are formatted correctly (quotation marks for text values, no quotes for real values...).


#### e. Table Name Replacement
Replaces `table` placeholder with the real table identifier from dataset.

---

## Evaluation
For human_readable_output and runnable_output methods, model selection relied on `rouge2`. 
For structured_output method the main metric was `logic_form_accuracy` (exact field match).
Final evaluation used execution accuracy, wether generated SQL returned the correct results.


With 74.03% Execution accuracy (using `flan-t5-small` with runnable output generation) we beat the original Seq2SQL 
model's evaluation from paper (still note that we did not evaluate on the same dataset)

---
|Model|Method | Execution accuracy| Exact Match accuracy| ROUGE2    | Model path |
|-----|-------|-------------------|---------------------|-----------|---|
|flan-t5-small| Human readable output |71.32%| **59.1%**| 90.7%     |`"/content/drive/MyDrive/SQLCoder/models/flan-t5-small-human-readable-output-no-sample-3/"`|
|flan-t5-small| Human readable output with col separators |70.66%| 58.3%|  70.3%    |`"/content/drive/MyDrive/SQLCoder/models/flan-t5-small-human-readable-output-no-sample-4/"`|
|flan-t5-base| Human readable output |**78.6%**| 71.8%| **96.7%** |`"/content/drive/MyDrive/SQLCoder/models/flan-t5-base-human-readable-output-no-sample/"`|
|flan-t5-small| Runnable output |**74.03%**|58.7%| **95.7%** |`"/content/drive/MyDrive/SQLCoder/models/flan-t5-small-runnable-output-no-sample/"`|
|flan-t5-small| Structured output |71.58%|X|   X       |`"/content/drive/MyDrive/SQLCoder/models/flan-t5-small-structured-output-no-sample/`|


---
Model used : `flan-t5-small`
Model saved at : `"/content/drive/MyDrive/SQLCoder/models/flan-t5-small-human-readable-output-no-sample-3/`

**Evaluation Details**
```
{'eval_loss': 0.1485850065946579,
 'eval_model_preparation_time': 0.0038,
 'eval_rouge1': 0.9163832743491616,
 'eval_rouge2': 0.90715059242323,
 'eval_rougeL': 0.9133319968723876,
 'eval_rougeLsum': 0.9133428603920335,
 'eval_exact_match_accuracy': 0.5910064239828693,
 'eval_runtime': 234.7387,
 'eval_samples_per_second': 67.641,
 'eval_steps_per_second': 1.061,
 'epoch': 5.0}

Detected 223 / 15878 (1.40 %) invalid prediction queries
 ```

---
Model used : `flan-t5-base`
Model saved at : `"/content/drive/MyDrive/SQLCoder/models/flan-t5-base-human-readable-output-no-sample/best_model`

**Evaluation Details**
 ```
{'eval_loss': 0.07574507594108582,
 'eval_rouge1': 0.987211249002752,
 'eval_rouge2': 0.9670818268511816,
 'eval_rougeL': 0.9707151267838618,
 'eval_rougeLsum': 0.9707108815327377,
 'eval_exact_match_accuracy': 0.7186673384557248,
 'eval_runtime': 813.7504,
 'eval_samples_per_second': 19.512,
 'eval_steps_per_second': 1.22,
 'epoch': 5.0}

Detected 120 / 15878 (0.76 %) invalid prediction queries
 ```

 ---
Model used : `flan-t5-small`

Model saved at : `"/content/drive/MyDrive/SQLCoder/models/flan-t5-small-runnable-output-no-sample/`

**Evaluation Details**
 ```

{'eval_loss': 0.07058710604906082,
 'eval_model_preparation_time': 0.0036,
 'eval_rouge1': 0.9743148548480041,
 'eval_rouge2': 0.9574181373365944,
 'eval_rougeL': 0.9598698491178133,
 'eval_rougeLsum': 0.9598857224270205,
 'eval_exact_match_accuracy': 0.5870386698576646,
 'eval_runtime': 438.6542,
 'eval_samples_per_second': 36.197,
 'eval_steps_per_second': 0.568,
 'epoch': 5.0}

Detected 811 / 15878 (5.11%) invalid prediction queries
 ```

 ---
Model used : `flan-t5-small`
Model saved at : `"/content/drive/MyDrive/SQLCoder/models/flan-t5-small-structured-output-no-sample/`

**Evaluation Details**
 ```

{'eval_loss': 0.06042969599366188,
 'eval_model_preparation_time': 0.0037,
 'eval_logic_form_accuracy': 0.5593273712054415,
 'eval_incorrect_json_ratio': 0.0,
 'eval_sel_accuracy': 0.9339967250283411,
 'eval_agg_accuracy': 0.8952638871394382,
 'eval_cond_col_accuracy': 0.7001698884246292,
 'eval_cond_op_accuracy': 0.6138941181872446,
 'eval_cond_value_accuracy': 0.6931906882776987,
 'eval_runtime': 751.7541,
 'eval_samples_per_second': 21.121,
 'eval_steps_per_second': 0.661,
 'epoch': 5.0}

Detected 272 / 15878 (1.71 %) invalid prediction queries
 ```

>Note that some asian characters are in the DB, thus needing to be in the tokenizer vocab, we could add these characters in our vocabulary.
>Furthermore maybe try upper/lower combinations for values in WHERE conditions when testing would make us win some performance points but would lead to additional computational time.
>Finally, using a code-specialized model such as CodeT5 or Codex could also help improve performance or reduce code complexity in pre and post-processing.

## Discussions
One of the main struggle I encountered during the project was dealing with the formats o columns and their values.
Some columns are of type `real` but contain values with `,` as thousands separator (e.g. `1,234`). Others are "text" but contain numeric values.
Finally most columns have names with special characters (commas, slashes, hyphens, parentheses...) that make it hard for the post-processing step. 
That was one of the main reasons why I decided to use similarity search for columns when post-processing and also 
why I tried to train a model that could generate directly runnable SQL queries from the table schema including types.

Overall, the `runnable_output` method seems promising with FLAN-T5 small model as it manages to  
outperform the original Seq2SQL model's performances and even most bigger LLMs performances in few-shot approaches without fine-tuning
(cf. [LLMSQL: Upgrading WikiSQL for the LLM Era of Text-to-SQL](https://arxiv.org/html/2510.02350v1) paper). However, 
one should keep in mind that the evaluation dataset is not exactly the same as the original WikiSQL dataset or the one used in the LLMSQL paper.

Moreover, I found that the `human_readable_output` method with SQLSanitizer post-processing also gives good results and could be a
good alternative to the `runnable_output` method if one wants to have more control over the post-processing step.

Finally, I think the whole post-processing step could be better thought out to ensure more robustness and better error handling.


## Future Work
In future work, I would explore:
* Add sample rows in the input prompt to give the model more context about the table data. 
* Experimenting with code-specialized models like CodeT5 or Codex for potentially better SQL generation.
* Experimenting with bigger models like FLAN-T5 base/large or XL to see if performance improves.
* Enhancing the SQLSanitizer with more robust type handling and error correction.
* Incorporating advanced string similarity algorithms (e.g., Levenshtein distance) for improved column matching and fine tune threshold.
* Clean dataset from SQL queries that result in empty results to avoid confusing the model during training.
* Include types in `sanitize_where_conditions()` method to ensure values are formatted correctly (quotation marks for text values, no quotes for real values...).