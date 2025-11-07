from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from datasets import Dataset
from dataclasses import dataclass
from config import BATCH_SIZE, EPOCHS, GENERATION_MAX_LENGTH
from model.utils import load_model_and_tokenizer, load_model_from_checkpoint
from evaluation.metrics import compute_metrics, compute_human_readable_metrics


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
        self.metrics = self._get_metrics()

    def _get_metrics(self):
        str_o_metrics = lambda x: compute_metrics(x, self.tokenizer)
        hro_metrics = lambda x: compute_human_readable_metrics(x, self.tokenizer)
        return str_o_metrics if self.config.mode == "structured_output" else hro_metrics

    def _load_model(self):
        self.model, self.tokenizer, self.data_collator = load_model_and_tokenizer(self.model_name)

    def load_model_from_checkpoint(self, model_checkpoint: str):
        self.model, self.tokenizer, self.training_args = load_model_from_checkpoint(model_checkpoint)


def load_trainer(text2sql_model: Text2SQL, train_dataset: Dataset, val_dataset: Dataset) -> Seq2SeqTrainer:
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

    trainer = Seq2SeqTrainer(
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
