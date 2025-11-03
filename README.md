# Senior Machine Learning Engineer Challenge - SQLCoder

## Overview

Welcome to the challenge for the Senior Machine Learning Engineer position in Natural Language Processing (NLP). 
This challenge is designed to assess your scientific reasoning, software engineering skills, and your ability to tackle practical NLP problems creatively.

You will have **one week (7 days)** from the moment you receive this document to complete the challenge. 
We encourage you to explore the problem as deeply as you can within this timeframe. 
The only constraint is that you should **not spend any money** (e.g., on paid APIs or cloud services) while solving the challenge.

If you do not have access to a GPU on your personal machine, you can use a **Kaggle notebook** or a Google Colab, which provide a limited amount of free GPU resources. Detailed instructions for using Kaggle are included below.

## Objective

Your task is to develop an **SQLCoder**: a system that converts natural language queries into executable SQL queries. 
Specifically, you will deliver a DNN model to perform this conversion and implement a minimal application that can run these queries against the provided database (in the data/ folder of this repo).

## Dataset

You have to use the provided version of the WikiSQL dataset for training and testing your model. 
This dataset is a pre-processed version of [WikiSQL dataset](https://huggingface.co/datasets/Salesforce/wikisql), and was originally published with the [SEQ2SQL paper](https://arxiv.org/pdf/1709.00103).
In this dataset, each natural language question is paired with a corresponding SQL query.

**Important notes on the dataset:**

The "human\_readable" SQL queries present in the dataset are logically correct but not immediately functional.
For example:

```sql
SELECT Notes FROM table WHERE Current slogan = SOUTH AUSTRALIA
```

This query contains the placeholder `table` and lacks proper formatting for values.

A functional version of the query would look like:

```sql
SELECT `Notes` FROM table_1_1000181_1 WHERE `Current slogan` = "SOUTH AUSTRALIA"
```

Note the explicit table name and the quotation marks.

Your model does not necessarily have to generate functionally correct queries directly. 
However, having identified or implemented strategies to generate executable SQL queries (directly from the model prediction or via post-processing) will be considered a strong plus.

**Important notes on the provided data:**

In the `data/` folder of this repository you will find 2 sub-folders.

* the dataset_files/ folder contains .jsonl files with the actual WikiSQL dataset samples (user question, info about the associated table, SQL query, etc.)
* The db_files/ folder contains the corresponding SQL databases

Please note that the provided DB is not exactly equivalent to the original WikiSQL database.
This is because the tables rows have been modified in order to match exactly the SQL queries in human-readable form. In other words, each "human_readable" SQL query is guaranteed to return the correct match on the provided db file if, as in the example above: 
* the correct values are enclosed in the correct quotation marks.
* the correct table name is used in place of the `table` placeholder. 

No other modification of the SQL queries is required. 


## Task Requirements

### 1. Model Development

* Training from scratch or fine-tuning a pre-trained model, transformer-based or not, etc. You are free to choose whatever architecture/training strategy you want !
* Train the model(s) on the WikiSQL dataset for the task of natural language to SQL conversion.
* You are free to use part or all of the data.
* You are free to use models already tailored to this task as baselines or for comparison, but we expect at least one original training/fine-tuning effort.
* This is your chance to shine and show your creativity !

### 2. Codebase and Submission 

Your final submission should include:

  * An archive containing all source code, scripts, and necessary instructions to run your solution.
  * A written report detailing how you approached the task, how you designed and evaluated the whole system, similar to what you would write to share with colleagues in a research lab. 
You should also talk about what could be added with more time.

### 3. Application Implementation (Extra)

* We would value if you could additionally develop a minimal application that can:

  * Take a natural language query from the user.
  * Convert it to an SQL query using your trained model.
  * Execute the query against the provided database and return the results.

* Tip: you do not need a working prediction model to develop the application. 


## Evaluation Criteria

* **Code quality:** Clarity, structure, and adherence to best practices.
* **Scientific rigor:** Depth of reasoning, exploration of ideas, and soundness of methodology.
* **Creativity:** Approaches to solving the problem and handling dataset challenges.
* **Problem understanding:** Ability to clearly identify and address the practical issues in transforming human-readable SQL to executable queries.
* **End-to-end system:** A working demonstration that connects all parts of your solution with a strategy to generate executable SQL queries will be considered a strong plus.   

The final model’s accuracy or performance on the database is not at all our primary evaluation criterion. 
Even if your model does not succeed in executing correct queries, an insightful, well-documented, and scientifically sound approach will be highly valued.

## Constraints

* You may only use free resources; do not spend any money on paid services or tools.
* You have 7 days to complete and submit the challenge.
* Share your submission as an archive, and include your written report as part of it.

## Good luck!

We look forward to seeing your creativity, engineering skills, and scientific approach shine through your submission.


## Appendix: Kaggle Usage Instructions

If you do not have a local GPU, you could use [Kaggle notebooks](https://www.kaggle.com/code) to run your training. Kaggle provides 30 hours per week of free GPU (P100) usage. To unlock GPU usage, you need to have a registered and verified account. 

### Important Notes

* Kaggle notebooks run inside Docker containers with many pre-installed Python packages. The latest Docker image details are here: [Kaggle Docker Python](https://github.com/Kaggle/docker-python).
* List of pre-installed dependencies is here: [kaggle\_requirements.txt](https://github.com/Kaggle/docker-python/blob/main/kaggle_requirements.txt).
* When using Python 3.11, most standard ML code should work without additional configuration.
* We recommend testing your code locally on small-scale experiments first to make the most of your free GPU time.

### Steps

1. Have your repository hosted on GitHub.
2. Generate a personal access token in GitHub with at least **Content: read** permissions. Copy this token securely (you won’t be able to see it again).
3. Enable GPU acceleration on your Kaggle notebook (you get 30 hours of P100 usage per week).
4. Enable Internet in your Kaggle notebook:

   * Click on the sidebar button at the bottom right corner.
   * Go to **Session Options**.
   * Toggle on **Internet**.
5. In your Kaggle notebook, click on **Add-ons** and add a secret:

   * Provide a label (e.g., `GITHUB_TOKEN`).
   * Paste your GitHub token in the value field.
6. Add the following code snippet to your notebook:

```python
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("GITHUB_TOKEN")

!git clone --branch <branch> https://<username>:{secret_value_0}@github.com/<username>/<repo>.git
```

7. Verify the repository was cloned successfully with:

```bash
!ls
```

8. Move into the repository directory:

```python
%cd repo
```

9. You can run any script by prepending `!`, for example:

```python
!python script.py
```

There should be no need to install additional dependencies since the Kaggle Docker image includes most commonly used ML libraries. This should be sufficient for a proof-of-concept like the one required for this challenge.