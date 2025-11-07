import os.path
from transformers import AutoTokenizer
from datasets import load_from_disk

from data.loader import load_dataset_from_json
from sanitizer.sql_sanitizer import SQLSanitizer
from utils import dict2query
from data.process import Processor
from config import DATA_PATH


class WikiSQLLoader:
    """Loader that loads data from the WikiSQL dataset."""

    def __init__(self, tokenizer: AutoTokenizer, data_path: str = DATA_PATH, mode="human_readable_output",
                 with_samples: bool = False):
        self.data_path = data_path
        self.sanitizer = SQLSanitizer()
        self.mode = mode
        assert mode in ["human_readable_output", "structured_output"], "Invalid mode. Choose either 'human_readable_output' or 'structured_output'."
        self.tokenizer = tokenizer
        self.with_samples = with_samples
        self.processor = Processor(tokenizer, mode=mode, with_samples=with_samples)

    def _check_existing_processed_data(self):
        if os.path.exists(f"{self.data_path}/processed/wikisql_train_{self.mode}_{'with_samples' if self.with_samples else 'no_samples'}") and \
           os.path.exists(f"{self.data_path}/processed/wikisql_val_{self.mode}_{'with_samples' if self.with_samples else 'no_samples'}") and \
           os.path.exists(f"{self.data_path}/processed/wikisql_test_{self.mode}_{'with_samples' if self.with_samples else 'no_samples'}"):
            return True
        return False

    def _load_existing_processed_data(self):
        train_data = load_from_disk(f"{self.data_path}/processed/wikisql_train_{self.mode}_{'with_samples' if self.with_samples else 'no_samples'}")
        val_data = load_from_disk(f"{self.data_path}/processed/wikisql_val_{self.mode}_{'with_samples' if self.with_samples else 'no_samples'}")
        test_data = load_from_disk(f"{self.data_path}/processed/wikisql_test_{self.mode}_{'with_samples' if self.with_samples else 'no_samples'}")
        return train_data, val_data, test_data

    def load_data(self):
        """Load and return data from the WikiSQL dataset."""
        # Check for existing processed data
        if self._check_existing_processed_data():
            print("Loading existing processed datasets from disk...")
            return self._load_existing_processed_data()
        else:
            print("No existing processed datasets found. Processing raw data...")

        # Loading Dataset from JSONL files
        train_data, val_data, test_data = load_dataset_from_json(self.data_path)

        # Map valid SQL queries to the dataset "valid_sql" is a runnable SQL query
        print("Sanitizing 'train' GT SQL queries...")
        train_data = train_data.map(lambda x: {
            "valid_sql": self.sanitizer.sanitize(dict2query(x["table"]["id"], x["table"]["header"], x["sql"]),
                                                 x["table"])})
        print("Sanitizing 'val' GT SQL queries...")
        val_data = val_data.map(lambda x: {
            "valid_sql": self.sanitizer.sanitize(dict2query(x["table"]["id"], x["table"]["header"], x["sql"]),
                                                 x["table"])})
        print("Sanitizing 'test' GT SQL queries...")
        test_data = test_data.map(lambda x: {
            "valid_sql": self.sanitizer.sanitize(dict2query(x["table"]["id"], x["table"]["header"], x["sql"]),
                                                 x["table"])})

        # Tokenize input prompts + get attention masks for inputs and GT outputs
        print("Tokenizing 'train' data...")
        train_data = train_data.map(lambda x: self.processor.preprocess_data_row(x))
        print("Tokenizing 'val' data...")
        val_data = val_data.map(lambda x: self.processor.preprocess_data_row(x))
        print("Tokenizing 'test' data...")
        test_data = test_data.map(lambda x: self.processor.preprocess_data_row(x))

        # Saving HF Dataset
        print("Saving processed datasets to disk...")
        train_data.save_to_disk(f"{self.data_path}/processed/wikisql_train_{self.mode}_{'with_samples' if self.with_samples else 'no_samples'}")
        val_data.save_to_disk(f"{self.data_path}/processed/wikisql_val_{self.mode}_{'with_samples' if self.with_samples else 'no_samples'}")
        test_data.save_to_disk(f"{self.data_path}/processed/wikisql_test_{self.mode}_{'with_samples' if self.with_samples else 'no_samples'}")

        return train_data, val_data, test_data


# if __name__ == '__main__':
#     from src.config import Config
#
#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
#
#     # Load data
#     data_loader = WikiSQLLoader(data_path=DATA_PATH, tokenizer=tokenizer,
#                                 mode="human_readable_output", with_samples=False)
#     train_data, val_data, test_data = data_loader.load_data()
#     print(f"Train size: {len(train_data)} | Val size: {len(val_data)} | Test size: {len(test_data)}")
#     print("Train features: ", train_data.features)
