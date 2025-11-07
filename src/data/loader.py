from config import SQL_DB_PATH, DATA_PATH

from sqlalchemy import create_engine
import os
import json
from datasets import Features, Value, Sequence, load_dataset


def load_db_engines(sql_db_path: str = SQL_DB_PATH):
    # ------- TRAIN --------
    train_db_path = os.path.join(sql_db_path, "train.db")
    train_db_engine = create_engine(f"sqlite:///{train_db_path}")

    # ------- VAL --------
    val_db_path = os.path.join(sql_db_path, "val.db")
    val_db_engine = create_engine(f"sqlite:///{val_db_path}")

    # ------- TEST --------
    test_db_path = os.path.join(sql_db_path, "test.db")
    test_db_engine = create_engine(f"sqlite:///{test_db_path}")

    print("\n\t SQL database engines created successfully.")

    return train_db_engine, val_db_engine, test_db_engine


def normalize_jsonl(in_path: str, out_path: str):
    """Normalizing jsonl to match the original format of WikiSQL"""
    print(os.getcwd())
    with open(in_path, "r", encoding="utf-8") as f_in, open(out_path, 'w', encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            obj = json.loads(line)
            sql = obj.get("sql", {})
            conds = sql.get("conds", [])
            if len(conds) > 0 and isinstance(conds[0], list):
                col_index = []
                operator_index = []
                val_index = []
                for c in conds:
                    col_index.append(c[0])
                    operator_index.append(c[1])
                    val_index.append(c[2])
                sql["conds"] = {
                    "col": col_index,
                    "op": operator_index,
                    "val": val_index
                }
            elif len(conds) == 0:
                sql["conds"] = {
                    "col": [],
                    "op": [],
                    "val": []
                }

            obj["sql"] = sql
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_dataset_from_json(data_path: str = DATA_PATH):
    """Load the dataset from JSONL files after normalizing "sql" objects and return train, val, and test datasets."""
    features = Features({
        "phase": Value("int64"),
        "question": Value("string"),
        "table": {
            "caption": Value("string"),
            "header": Sequence(Value("string")),
            "id": Value("string"),
            "name": Value("string"),
            "page_id": Value("string"),
            "page_title": Value("string"),
            "rows": Sequence(Sequence(Value("string"))),
            "section_title": Value("string"),
            "types": Sequence(Value("string")),
        },
        "sql": {
            "human_readable": Value("string"),
            "sel": Value("int64"),
            "agg": Value("int64"),
            "conds": Features({
                "col": Sequence(Value("int64")),  # Index of column used in condition
                "op": Sequence(Value("int64")),  # Operator index mapping to SQL comparison operators
                "val": Sequence(Value("string")),  # Value used for comparison
            }),
        }
    })

    # ------------ TRAIN -------------
    print("\n\t Normalizing train set...")
    # Normalization
    train_input_path = os.path.join(data_path, "train.jsonl")
    train_output_path = os.path.join(data_path, "train_normalized.jsonl")
    normalize_jsonl(train_input_path, train_output_path)

    # ------------ VAL -------------
    print("\n\t Normalizing val set...")
    # Normalization
    val_input_path = os.path.join(data_path, "val.jsonl")
    val_output_path = os.path.join(data_path, "val_normalized.jsonl")
    normalize_jsonl(val_input_path, val_output_path)

    # ------------ TEST -------------
    print("\n\t Normalizing test set...")
    # Normalization
    test_input_path = os.path.join(data_path, "test.jsonl")
    test_output_path = os.path.join(data_path, "test_normalized.jsonl")
    normalize_jsonl(test_input_path, test_output_path)

    # ------------ DATA LOADING -----
    data = load_dataset("json",
                        data_files={
                            "train": train_output_path,
                            "val": val_output_path,
                            "test": test_output_path,
                        },
                        features=features)

    train_data = data["train"]
    val_data = data["val"]
    test_data = data["test"]

    print("\n\t Train, Val and Test datasets loaded successfully.")
    print("\t Train size:", len(train_data))
    print("\t Val size:", len(val_data))
    print("\t Test size:", len(test_data))

    return train_data, val_data, test_data


# if __name__ == '__main__':
#     # Loading SQL DB Engines
#     train_db_engine, val_db_engine, test_db_engine = load_db_engines()
#
#     # Loading Dataset
#     train_data, val_data, test_data = load_dataset_from_json()
