import argparse
from data.wiki_sql_loader import WikiSQLLoader
from model.text_to_sql import Text2SQL, Text2SQLConfig, load_trainer
from data.loader import load_db_engines
from config import DATA_PATH


def main():
    parser = argparse.ArgumentParser(description="Train a Text-to-SQL model")

    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--generation_max_length", type=int, default=128, help="Maximum generation length")
    parser.add_argument("--metric_for_best_model", type=str, default="rouge2", help="Metric used to select best model")
    parser.add_argument("--mode", type=str, default="human_readable_output", help="Output mode")
    parser.add_argument("--output_dir", type=str, default="./models/text2sql_hro_small", help="Output directory")

    args = parser.parse_args()

    # Load Text2SQL model and tokenizer
    config = Text2SQLConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        generation_max_length=args.generation_max_length,
        metric_for_best_model=args.metric_for_best_model,
        mode=args.mode,
        output_dir=args.output_dir,
    )
    text2sql_model = Text2SQL(config)

    # Load data
    data_loader = WikiSQLLoader(data_path=DATA_PATH, tokenizer=text2sql_model.tokenizer,
                                mode=text2sql_model.mode, with_samples=False)
    train_data, val_data, test_data = data_loader.load_data()
    print(f"Train size: {len(train_data)} | Val size: {len(val_data)} | Test size: {len(test_data)}")

    # Load trainer
    trainer = load_trainer(text2sql_model, train_data, val_data)

    # Train
    print("Training on device : ", trainer.args.device)
    print("With following config : ", config)
    trainer.train()

    # Evaluate
    trainer.evaluate(eval_dataset=test_data)

    # Evaluate execution accuracy
    train_db_engine, val_db_engine, test_db_engine = load_db_engines()
    trainer.evaluate_execution_accuracy(test_data, test_db_engine, mode=text2sql_model.mode)

    # Save model
    trainer.save_model()


if __name__ == '__main__':
    main()
