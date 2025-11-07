from model.text_to_sql import Text2SQL, Text2SQLConfig
from data.wiki_sql_loader import WikiSQLLoader
from config import DATA_PATH

if __name__ == '__main__':
    import os

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

    text2sql_model = Text2SQL(config)
    text2sql_model.load_model_from_checkpoint(model_checkpoint=model_checkpoint)

    # Load data
    data_loader = WikiSQLLoader(data_path=DATA_PATH, tokenizer=text2sql_model.tokenizer,
                                mode=text2sql_model.mode, with_samples=False)
    train_data, val_data, test_data = data_loader.load_data()

    questions = ['Tell me what the notes are for South Australia ',
                 'What is the current series where the new series began in June 2011?',
                 'What is the format for South Australia?']
    tables = [
        {
            'header': ['State/territory', 'Text/background colour', 'Format', 'Current slogan', 'Current series',
                       'Notes'],
            'id': '1-1000181-1',
        },
        {
            'header': ['State/territory', 'Text/background colour', 'Format', 'Current slogan', 'Current series',
                       'Notes'],
            'id': '1-1000181-1',
        },
        {
            'caption': '',
            'header': ['State/territory', 'Text/background colour', 'Format', 'Current slogan', 'Current series',
                       'Notes'],
            'id': '1-1000181-1'
        }

    ]

    sql_queries = text2sql_model.generate(questions, tables)
    for q, sql in zip(questions, sql_queries):
        print("Question: ", q)
        print("Generated SQL: ", sql)