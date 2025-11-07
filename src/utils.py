from sqlalchemy.engine.base import Engine
from sqlalchemy import text
import re
import json
from config import IDX2OP, AGGS_ORDERED, NEW_TOKENS, SPECIAL_TOKENS


def sql_engine(query: str, engine: Engine) -> str:
    """
    Allows to perform SQL queries on a table. Returns a string representation of the result.
    Args:
        query: The query to perform. This should be correct SQL.
        engine: The SQLAlchemy engine to use.
    """
    output = ""
    with engine.connect() as con:
        rows = con.execute(text(query))
        for row in rows:
            output += "\n" + str(row)
    return output


def quote_str(name: str) -> str:
    """Quotes a string for SQL query"""
    return "`" + name.replace("`", "``") + "`"


def quote_value(value: str) -> str:
    """Quotes a value for SQL query"""
    if value is None: return "NULL"
    if value.replace(" ", "").isdigit(): return value.replace(" ", "")
    return '"' + value.replace('"', '""') + '"'


def dict2query(table_id: str, headers: list, sql_dict: dict):
    """Takes generated/GT sql_dict to parse it to a valid executable SQL query given table_id and headers"""
    # Aggregate
    agg = AGGS_ORDERED[sql_dict.get("agg", 0)]
    agg = agg + " " if agg is not None else ""

    query = "SELECT " + agg + quote_str(headers[sql_dict["sel"]])

    # FROM
    table_name = "table_" + table_id.replace("-", "_")
    query += " FROM " + table_name

    # Conditions
    conds = sql_dict["conds"]
    n_conds = len(conds["col"])
    if n_conds > 0:
        query += " WHERE"

    for i in range(n_conds):
        col = quote_str(headers[conds["col"][i]])
        op = IDX2OP[conds["op"][i]]
        val = quote_value(conds["val"][i])

        if i < n_conds - 1:
            query += f" {col} {op} {val} AND"
        else:
            query += f" {col} {op} {val}"

    return query


def extract_json(text: str, special_tokens: dict = SPECIAL_TOKENS) -> dict:
    # Drop tokenizer special tokens
    text = strip_specials(text, special_tokens=special_tokens)

    # Drop braces, brackets
    text = unformat_string(text)

    m = re.search(r'\{.*\}', text, flags=re.S)
    if not m:
        # TODO: Fallback ?
        raise ValueError("No matching braces were found in JSON extraction...")
    json_str = m.group(0)

    try:
        return json.loads(json_str)
    except:
        raise ValueError("JSON loading failed")


def strip_specials(text: str, special_tokens: dict = SPECIAL_TOKENS) -> str:
    """
    special tokens map :  {'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>', 'additional_special_tokens': ['<LBRACE>', '<RBRACE>', '<LBRACK>', '<RBRACK>']}
    additional_tokens will be stripped by format_string()
    """
    to_strip = {v for v in special_tokens.values() if isinstance(v, str)}

    # Replace any number of these tokens (surrounded by optional whitespace) with a single space
    pattern = re.compile(r'(?:\s*(?:' + '|'.join(map(re.escape, to_strip)) + r')\s*)+')
    return pattern.sub(' ', text).strip()


def format_string(s: str) -> str:
    """Replaces brackets and braces to special tokens"""
    pattern = re.compile("|".join(map(re.escape, NEW_TOKENS)))
    return pattern.sub(lambda m: NEW_TOKENS[m.group(0)], s)


def unformat_string(s: str) -> str:
    """Replaces braces/brackets special tokens to brackets and braces"""
    REV_TOKENS = {v: k for k, v in NEW_TOKENS.items()}
    pattern = re.compile("|".join(map(re.escape, REV_TOKENS)))
    return pattern.sub(lambda m: REV_TOKENS[m.group(0)], s)
