# scripts/extract_schemas.py
import pandas as pd
from model.text_to_sql import Text2SQL, Text2SQLConfig
from data.wiki_sql_loader import WikiSQLLoader
from config import DATA_PATH

if __name__ == '__main__':
    model_checkpoint = "./models/flan-t5-small-hro-no-sample"
    config = Text2SQLConfig(
        model_name="google/flan-t5-small",
        batch_size=16,
        num_train_epochs=5,
        generation_max_length=128,
        metric_for_best_model="rouge2",
        mode="human_readable_output",
        output_dir="./models/text2sql_hro_small"
    )

    model = Text2SQL(config)
    model.load_model_from_checkpoint(model_checkpoint=model_checkpoint)

    # Load only once
    data_loader = WikiSQLLoader(
        data_path=DATA_PATH,
        tokenizer=model.tokenizer,
        mode=model.mode,
        with_samples=False
    )
    _, _, test_data = data_loader.load_data()

    # Extract schemas
    records = [
        {
            "id": t["table"].get("id", ""),
            "caption": t["table"].get("caption", ""),
            "header": t["table"].get("header", []),
            "types": t["table"].get("types", []),
        }
        for t in test_data
    ]

    df = pd.DataFrame(records)
    df.to_pickle("../data/test_schemas.pkl")
    print(f"Saved {len(df)} table schemas to data/test_schemas.pkl")