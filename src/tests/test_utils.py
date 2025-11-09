import unittest
from utils import quote_str, quote_value, dict2query, strip_specials, format_backtick, format_string, unformat_string, \
    extract_json
from config import SPECIAL_TOKENS, NEW_TOKENS


# Run in src/ python -m unittest discover
# To run all tests
class TestUtils(unittest.TestCase):

    def test_quote_str(self):
        self.assertEqual(quote_str("name"), "`name`")
        self.assertEqual(quote_str("user`data"), "`user``data`")  # backtick escaping

    def test_quote_value_strings(self):
        self.assertEqual(quote_value("hello"), '"hello"')  # add quotes
        self.assertEqual(quote_value('"already_quoted"'), '"already_quoted"')
        self.assertEqual(quote_value(" world "), '"world"')  # trims whitespace

    def test_dict2query_basic(self):
        sql_dict = {
            "agg": 0,  # None → no aggregation
            "sel": 1,
            "conds": {"col": [], "op": [], "val": []}
        }
        headers = ["id", "name"]
        types = ["real", "text"]

        query = dict2query("table_123", headers, sql_dict, types)
        self.assertEqual(query, "SELECT `name` FROM table_123")

    def test_dict2query_with_agg_and_cond(self):
        sql_dict = {
            "agg": 1,  # MAX
            "sel": 0,
            "conds": {
                "col": [1, 2],
                "op": [0, 1],  # "=" and ">"
                "val": ["john", "10"]
            }
        }
        headers = ["age", "name", "score"]
        types = ["real", "text", "real"]

        query = dict2query("table_abc_xyz", headers, sql_dict, types)

        expected = (
            "SELECT MAX(`age`) FROM table_abc_xyz "
            "WHERE `name` = \"john\" AND `score` > 10"
        )
        self.assertEqual(query, expected)

    def test_dict2query_with_no_replace_table(self):
        sql_dict = {
            "agg": 4,  # SUM
            "sel": 0,
            "conds": {"col": [0], "op": [2], "val": ["1,000"]}
        }
        headers = ["amount"]
        types = ["real"]

        query = dict2query("<table>", headers, sql_dict, types)
        self.assertEqual(query, "SELECT SUM(`amount`) FROM <table> WHERE `amount` < 1000")

    def test_strip_specials_removes_tokens(self):
        text = "Hello <pad> world </s> <unk>"
        result = strip_specials(text, special_tokens=SPECIAL_TOKENS)
        self.assertEqual(result, "Hello world")

    def test_strip_specials_keeps_normal_text(self):
        text = "No special tokens here"
        self.assertEqual(strip_specials(text, special_tokens=SPECIAL_TOKENS), text)

    def test_format_backtick(self):
        s = "some `text` with backticks"
        self.assertEqual(format_backtick(s), "some [backtick]text[backtick] with backticks")

    def test_format_string(self):
        s = "{key: [1, 2]}"
        expected = "[LBRACE]key: [LBRACK]1, 2[RBRACK][RBRACE]"
        self.assertEqual(format_string(s), expected)

    def test_unformat_string(self):
        s = "[LBRACE]key: [LBRACK]1, 2[RBRACK][RBRACE]"
        expected = "{key: [1, 2]}"
        self.assertEqual(unformat_string(s), expected)

    def test_unformat_string_with_backtick(self):
        s = "[LBRACE][backtick]key[backtick]: 1[RBRACE]"
        expected = "{`key`: 1}"
        self.assertEqual(unformat_string(s), expected)

    def test_extract_json_with_specials_and_tokens(self):
        text = "</s> <unk> [LBRACE]\"a\": 5[RBRACE] <pad>"
        result = extract_json(text, special_tokens=SPECIAL_TOKENS)
        self.assertEqual(result, {"a": 5})

    def test_extract_json_no_braces_raises(self):
        text = "no braces here"
        with self.assertRaises(ValueError):
            extract_json(text)

    def test_extract_json_invalid_json_raises(self):
        text = "{invalid json}"
        with self.assertRaises(ValueError):
            extract_json(text)


if __name__ == '__main__':
    unittest.main()
