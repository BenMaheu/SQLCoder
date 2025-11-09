import os
from pathlib import Path

import streamlit as st
import pandas as pd
from model.text_to_sql import Text2SQL, Text2SQLConfig
from data.loader import load_db_engines
from utils import run_sql_query

MODEL_OPTIONS = {
    "human_readable_output": "./models/flan-t5-small-hro-no-sample",
    "structured_output": "./models/flan-t5-small-structured-output-no-sample",
    "runnable_output": "./models/flan-t5-small-runnable-output-no-sample",
}


# Loading model at init
@st.cache_resource
def load_model(model_mode: str):
    model_checkpoint = MODEL_OPTIONS[model_mode]
    config = Text2SQLConfig(
        model_name="google/flan-t5-small",
        batch_size=16,
        num_train_epochs=5,
        generation_max_length=128,
        metric_for_best_model="rouge2",
        mode=model_mode,
        output_dir="./models/flan-t5-small-structured-output-no-sample"
    )
    model = Text2SQL(config)
    model.load_model_from_checkpoint(model_checkpoint=model_checkpoint)
    print("Loaded model from checkpoint.")
    return model


@st.cache_data
def load_table_schemas():
    path = os.path.join(Path(os.getcwd()).parent, "data/test_schemas.pkl")
    df = pd.read_pickle(path)

    # Precompute the labels once
    df["label"] = df.apply(
        lambda r: f"{r['id']} — {r['caption']}" if r["caption"] else r["id"], axis=1
    )

    df = df.drop_duplicates(subset=["label"]).reset_index(drop=True)
    return df


# Interface
def main():
    st.title("SQLCoder Demo App")
    st.write("Ask a natural language question on your database and generate the corresponding SQL query.")


    selected_model_name = st.selectbox("Select a model:", list(MODEL_OPTIONS.keys()))
    st.info(f"Currently using model: **{selected_model_name}**")

    # Load model and data
    text2sql_model = load_model(selected_model_name)
    # Load table schemas
    df_tables = load_table_schemas()
    # Load SQL database engine
    _, _, test_db_engine = load_db_engines()  # à adapter à ton backend SQL


    # Table selection
    selected_label = st.selectbox("Select a table:", df_tables["label"])
    selected_table = df_tables[df_tables["label"] == selected_label].iloc[0]

    st.subheader("Selected DB Schema:")
    st.dataframe(pd.DataFrame({"Columns": selected_table["header"]}))

    # User's question
    user_question = st.text_input("Ask a question :",
                                  placeholder="Ex: What clu was in toronto 1995-96?, What is terrence ross' nationality?, Which club was in toronto 2003-06?")

    if st.button("Generate SQL query"):
        if not user_question:
            st.warning("Ask a question before generating a query.")
        else:
            with st.spinner("Generating SQL query..."):
                sql_queries, raw_preds = text2sql_model.generate([user_question], [selected_table], return_raw=True)
                st.session_state["sql_query"], st.session_state["raw_pred"] = sql_queries[0], raw_preds[0]

    if "raw_pred" in st.session_state:
        st.text_area("Raw model output:", st.session_state["raw_pred"])

    if "sql_query" in st.session_state:
        st.text("Sanitized SQL query:")
        st.code(st.session_state["sql_query"], language="sql")

    if "sql_query" in st.session_state:
        # Exécution de la requête
        if st.button("Run SQL query on db (test_sql_db)"):
            try:
                results = run_sql_query(st.session_state["sql_query"], test_db_engine)
                print("Results : ", results)
                st.success("Query successfully run.")
                st.text("Query Results:")
                st.code(results, language="sql")
            except Exception as e:
                st.error(f"Erreur lors de l'exécution de la requête : {e}")


if __name__ == "__main__":
    main()
