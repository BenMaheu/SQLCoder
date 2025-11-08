from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from datasets import Dataset
import torch
from typing import List, Dict, Union, Tuple
from tqdm import tqdm
from dataclasses import dataclass
import re

from data.process import Processor
from config import BATCH_SIZE, EPOCHS, GENERATION_MAX_LENGTH, ALLOWED_MODES
from model.utils import load_model_and_tokenizer, load_model_from_checkpoint
from evaluation.metrics import compute_metrics, compute_human_readable_metrics
from sanitizer.sql_sanitizer import SQLSanitizer
from utils import run_sql_query, extract_json, unformat_string, strip_specials


@dataclass
class Text2SQLConfig:
    model_name: str = "google/flan-t5-small"
    batch_size: int = BATCH_SIZE
    num_train_epochs: int = EPOCHS
    generation_max_length: int = GENERATION_MAX_LENGTH
    metric_for_best_model: str = "rouge2"
    mode: str = "human_readable_output"  # or "structured_output"
    output_dir: str = "./models/text2sql_model"

    def __repr__(self):
        text = "\n"
        for attr in self.__dataclass_fields__:
            text += f"\t{attr}: {getattr(self, attr)}" + "\n"
        return text


class Text2SQL:
    def __init__(self, config: Text2SQLConfig):
        self.model_name = config.model_name
        self.trainer = None
        self.config = config
        self.mode = config.mode
        self._load_model()
        self.device = self.model.device
        self.metrics = self._get_metrics()

    def _get_metrics(self):
        str_o_metrics = lambda x: compute_metrics(x, self.tokenizer)
        hro_metrics = lambda x: compute_human_readable_metrics(x, self.tokenizer)
        return str_o_metrics if self.config.mode == "structured_output" else hro_metrics

    def _load_model(self):
        self.model, self.tokenizer, self.data_collator = load_model_and_tokenizer(self.model_name)

    def load_model_from_checkpoint(self, model_checkpoint: str):
        self.model, self.tokenizer, self.training_args = load_model_from_checkpoint(model_checkpoint)

    def generate(self, questions: List[str],
                 tables: List[Dict],
                 max_new_tokens: int = GENERATION_MAX_LENGTH,
                 num_beams=4, return_raw=False, with_samples=False, verbose=0) -> Union[List[str], Tuple[List[str], List[str]]]:
        # Preprocess questions (format prompts)
        processor = Processor(self.tokenizer, mode=self.mode, with_samples=with_samples)
        formatted_questions = [processor.get_input_prompt({"question": q, "table": table}) for q, table in
                               zip(questions, tables)]

        with torch.no_grad():
            model_inputs = self.tokenizer(formatted_questions, return_tensors="pt", truncation=True, padding=True,
                                          max_length=max_new_tokens).to(self.device)
            outputs_logits = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
            preds = self.tokenizer.batch_decode(outputs_logits, skip_special_tokens=False)

        sanitized_preds = []
        raw_preds = []
        sanitizer = SQLSanitizer()
        for raw_pred_str, table in zip(preds, tables):
            # Sanitizing prediction
            if self.mode == "structured_output":
                try:
                    sanitized_query = sanitizer.sanitize(extract_json(raw_pred_str), table, verbose=verbose)
                except:
                    try:
                        sanitized_query = extract_json(raw_pred_str)
                    except:
                        sanitized_query = raw_pred_str
            else:
                sanitized_query = sanitizer.sanitize(raw_pred_str, table, verbose=verbose)

            sanitized_preds.append(sanitized_query)
            raw_preds.append(raw_pred_str)
        if return_raw:
            return sanitized_preds, raw_preds
        return sanitized_preds


class Text2SQLTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate_execution_accuracy(self, eval_dataset: Dataset, eval_db_engine, mode="human_readable_output"):
        assert mode in ALLOWED_MODES, f"Invalid mode. Choose in {ALLOWED_MODES}."
        sanitizer = SQLSanitizer()
        execution_accuracy = 0
        invalid_san_queries = []  # Nb of queries generated that did not pass sanitizing
        invalid_sql_queries = []
        total = 0

        for batch in tqdm(batchify(eval_dataset, BATCH_SIZE),
                          total=len(eval_dataset) // BATCH_SIZE + 1):
            prompts = batch["input"]
            gold_sqls = batch["valid_sql"]

            # Per-batch generation
            with torch.no_grad():
                model_inputs = self.tokenizer(prompts, return_tensors="pt", truncation=True, padding=True,
                                              max_length=128).to(self.model.device)
                outputs = self.model.generate(**model_inputs, max_new_tokens=128, num_beams=4)
                preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

            for prompt, gold_sql, pred_sql, table in zip(prompts, gold_sqls, preds, batch["table"]):
                # Try to execute gold on db if fails -> skip
                try:
                    gold_result = run_sql_query(gold_sql, eval_db_engine)
                except Exception as e:
                    print(f"Skipping gold sql that can't run on db : {gold_sql}")
                    print(e)
                    print("\n")
                    continue

                total += 1
                try:
                    if mode == "human_readable_output":
                        # Sanitize raw prediction
                        sanitized_query = sanitizer.sanitize(pred_sql, table)
                    elif mode == "structured_output":
                        sanitized_query = sanitizer.sanitize(extract_json(pred_sql), table)
                    elif mode == "runnable_output":
                        # Replace <table> placeholder by the correct table id
                        table_id = table["id"]
                        table_id = "table_" + table_id.replace("-", "_")
                        pred_sql = pred_sql.replace("<table>", table_id)
                        pred_sql = re.sub("<table>", table_id, pred_sql)
                        pred_sql = unformat_string(strip_specials(pred_sql))
                except:
                    invalid_san_queries.append({"pred_sql": pred_sql})
                    continue

                # Compare SQL results
                # Both sanitize work
                try:
                    pred_result = run_sql_query(sanitized_query, eval_db_engine)
                except Exception as e:
                    print("\n---Invalid sanitized pred query---")
                    print("Prompt : ", prompt)
                    print("Raw Pred : ", pred_sql)
                    print("Sanitized Pred SQL : ", sanitized_query)
                    print("Gold SQL : ", gold_sql)

                    invalid_sql_queries.append({
                        "prompt": prompt,
                        "pred_sql": pred_sql,
                        "sanitized_gold_query": gold_sql,
                        "sanitized_query": sanitized_query,
                    })
                    continue

                if pred_result == gold_result:
                    execution_accuracy += 1

        print("Execution accuracy : {:.2f}%".format(execution_accuracy / (1e-12 + total) * 100))
        return {
            "execution_accuracy": execution_accuracy / (1e-12 + total),
        }


def load_trainer(text2sql_model: Text2SQL, train_dataset: Dataset, val_dataset: Dataset) -> Text2SQLTrainer:
    training_args = Seq2SeqTrainingArguments(
        output_dir=text2sql_model.config.output_dir,
        per_device_train_batch_size=text2sql_model.config.batch_size,  # 32/64 for small, 16 for base
        num_train_epochs=text2sql_model.config.num_train_epochs,
        per_device_eval_batch_size=text2sql_model.config.batch_size,  # 32/64 for small, 16 for base
        predict_with_generate=True,
        eval_strategy="epoch",
        do_train=True,
        do_eval=True,
        logging_steps=500,
        save_strategy="epoch",
        generation_max_length=text2sql_model.config.generation_max_length,
        # save_steps=1000,
        # eval_steps=1000,
        overwrite_output_dir=True,
        save_total_limit=3,
        load_best_model_at_end=True,
        push_to_hub=False,
        metric_for_best_model=text2sql_model.config.metric_for_best_model,
        greater_is_better=True,
        # fp16=True,
    )

    trainer = Text2SQLTrainer(
        model=text2sql_model.model,
        args=training_args,
        compute_metrics=text2sql_model.metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=text2sql_model.tokenizer,
        data_collator=text2sql_model.data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Sanity check
    print("\n\n\n\tSanity Check: Evaluating on a small subset of validation data")
    trainer.evaluate(eval_dataset=val_dataset.select(range(20)))

    return trainer


def batchify(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]

# if __name__ == '__main__':
#     from src.data.wiki_sql_loader import WikiSQLLoader
#     from config import DATA_PATH
#
#     # Load Text2SQL model and tokenizer
#     config = Text2SQLConfig(model_name="google/flan-t5-small",
#                             batch_size=16,
#                             num_train_epochs=5,
#                             generation_max_length=128,
#                             metric_for_best_model="rouge2",
#                             mode="human_readable_output",
#                             output_dir="./models/text2sql_hro_small")
#     text2sql_model = Text2SQL(config)
#
#
#     # Load data
#     data_loader = WikiSQLLoader(data_path=DATA_PATH, tokenizer=text2sql_model.tokenizer,
#                                 mode=text2sql_model.mode, with_samples=False)
#     train_data, val_data, test_data = data_loader.load_data()
#     print(f"Train size: {len(train_data)} | Val size: {len(val_data)} | Test size: {len(test_data)}")
#     print("Train features: ", train_data.features)
#
#     # Load trainer
#     trainer = load_trainer(text2sql_model, train_data, val_data)
#
#     # # Train
#     # trainer.train()
#     #
#     # # Evaluate
#     # eval_results = trainer.evaluate(eval_dataset=test_data)
