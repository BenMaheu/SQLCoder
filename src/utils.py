from sqlalchemy.engine.base import Engine
from sqlalchemy import text
import re
import json
from config import IDX2OP, AGGS_ORDERED, NEW_TOKENS, SPECIAL_TOKENS


def run_sql_query(query: str, engine: Engine) -> str:
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
    if re.match(r'^-?\d+(\.\d+)?$', value.strip()):
        # It's a number, leave as is
        return value
    else:
        # It's a string, ensure double quotes
        value = value.strip()
        if not (value.startswith('"') and value.endswith('"')):
            value = f'"{value}"'
        return value


def dict2query(table_name: str, headers: list, sql_dict: dict, types: list[str]) -> str:
    """
    Takes generated/GT sql_dict to parse it to a valid executable SQL query given table_id and headers

    :param table_name: exact table_id to be used in the FROM clause
    :param headers: list of headers/columns of the table
    :param sql_dict: sql dictionary from WikiSQL
    :param types: list of types of the columns
    :return: correct/runnable SQL query string with correct quotes
    :rtype:
    """
    # Aggregate
    agg = AGGS_ORDERED[sql_dict.get("agg", 0)]
    agg = agg + "(" if agg is not None else ""

    query = "SELECT " + agg + quote_str(headers[sql_dict["sel"]])
    query = query + ")" if len(agg) > 0 else query

    # FROM
    query += " FROM " + table_name

    # Conditions
    conds = sql_dict["conds"]
    n_conds = len(conds["col"])
    if n_conds > 0:
        query += " WHERE"

    for idx_cond in range(n_conds):
        col = quote_str(headers[conds["col"][idx_cond]])
        op = IDX2OP[conds["op"][idx_cond]]
        if types[conds["col"][idx_cond]] == "text":
            val = '"' + conds["val"][idx_cond] + '"'
        else:
            val = conds["val"][idx_cond]

        if idx_cond < n_conds - 1:
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
    """Replaces brackets, braces and backtick to special tokens"""
    pattern = re.compile("|".join(map(re.escape, NEW_TOKENS)))
    s = pattern.sub(lambda m: NEW_TOKENS[m.group(0)], s)
    return format_backtick(s)


def format_backtick(s: str) -> str:
    """Replaces backticks to special [backtick] token"""
    s = re.sub(r'`', '[backtick]', s)
    return s


def unformat_string(s: str) -> str:
    """Replaces braces/brackets special tokens to brackets and braces"""
    REV_TOKENS = {v: k for k, v in NEW_TOKENS.items()}
    pattern = re.compile("|".join(map(re.escape, REV_TOKENS)))
    s = pattern.sub(lambda m: REV_TOKENS[m.group(0)], s)

    # Replacing backtick token to actual backtick
    s = re.sub(r'\[backtick\]', '`', s)
    return s
