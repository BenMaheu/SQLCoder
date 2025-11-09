import unittest
import json
from unittest.mock import MagicMock
from data.process import Processor, get_gt_structured_output, get_gt_runnable_output, get_gt_human_readable_output, \
    format_input, format_input_with_type, format_structured_output

from config import COL_SEP, SAMPLE_SEP, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH, ALLOWED_MODES


class TestProcessor(unittest.TestCase):
    """Unit tests for Processor class and helpers."""

    def setUp(self):
        # Mock tokenizer (simulates a HF tokenizer)
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.side_effect = lambda x, **kwargs: {"input_ids": [1, 2, 3], "text": x}

        # Default Processor
        self.processor = Processor(
            tokenizer=self.mock_tokenizer,
            mode="human_readable_output",
            with_samples=False,
            column_sep=COL_SEP,
            samples_sep=SAMPLE_SEP,
            max_in_tokens=MAX_INPUT_LENGTH,
            max_out_tokens=MAX_OUTPUT_LENGTH,
        )

        self.example = {
            "question": "What is the capital of France?",
            "table": {
                "header": ["Country", "Capital"],
                "types": ["text", "text"],
            },
            "sql": {
                "human_readable": "SELECT Capital FROM table WHERE Country = France;",
                "agg": 0,
                "sel": 1,
                "conds": {"col": [0], "op": [0], "val": ["France"]},
            },
        }

    # --- Basic Initialization ---
    def test_processor_init(self):
        self.assertEqual(self.processor.mode, "human_readable_output")
        self.assertIn(self.processor.mode, self.processor.allowed_modes)
        self.assertEqual(self.processor.max_in_tokens, MAX_INPUT_LENGTH)

    # --- get_input_prompt ---
    def test_get_input_prompt_without_type(self):
        result = self.processor.get_input_prompt(self.example)
        self.assertIn("question:", result)
        self.assertIn("columns:", result)
        self.assertIn("Country", result)

    def test_get_input_prompt_with_type(self):
        self.processor.with_type = True
        result = self.processor.get_input_prompt(self.example)
        self.assertIn("types:", result)
        self.assertIn("text", result)

    # --- preprocess_data_row ---
    def test_preprocess_data_row_human_readable(self):
        out = self.processor.preprocess_data_row(self.example.copy(), test_mode=False)
        self.assertIn("labels", out)
        self.assertIn("input_ids", out)
        self.assertEqual(out["labels"], [1, 2, 3])

    def test_preprocess_data_row_structured_output(self):
        p = Processor(self.mock_tokenizer, mode="structured_output")
        out = p.preprocess_data_row(self.example.copy())
        self.assertIn("labels", out)
        self.assertEqual(out["labels"], [1, 2, 3])

    def test_preprocess_data_row_runnable_output(self):
        p = Processor(self.mock_tokenizer, mode="runnable_output")
        out = p.preprocess_data_row(self.example.copy())
        self.assertIn("labels", out)
        self.assertEqual(out["labels"], [1, 2, 3])

    def test_preprocess_data_row_test_mode(self):
        out = self.processor.preprocess_data_row(self.example.copy(), test_mode=True)
        self.assertIn("input_ids", out)

    def test_preprocess_data_row_invalid_mode_raises(self):
        p = Processor(self.mock_tokenizer, mode="human_readable_output")
        p.mode = "invalid_mode"
        with self.assertRaises(ValueError):
            p.preprocess_data_row(self.example.copy())

    # --- format_input ---
    def test_format_input(self):
        question = "Sample question"
        headers = ["h1", "h2"]
        text = format_input(question, headers)
        self.assertIn("task: text-to-sql", text)
        self.assertIn("columns:", text)

    # --- format_input_with_type ---
    def test_format_input_with_type(self):
        question = "Another question"
        headers = ["col1", "col2"]
        types = ["text", "number"]
        text = format_input_with_type(question, headers, types)
        self.assertIn("types:", text)
        self.assertIn("text", text)
        self.assertIn("number", text)

    # --- get_gt_structured_output ---
    def test_get_gt_structured_output(self):
        res = get_gt_structured_output(self.example)
        d = json.loads(res.replace("[LBRACE]", "{").replace("[RBRACE]", "}").replace("[LBRACK]", "[").replace("[RBRACK]", "]"))  # approximate unformat
        self.assertIn("sel", d)
        self.assertIn("agg", d)
        self.assertIn("conds", d)

    # --- get_gt_human_readable_output ---
    def test_get_gt_human_readable_output(self):
        self.assertEqual(
            get_gt_human_readable_output(self.example),
            "SELECT Capital FROM table WHERE Country = France;"
        )

    # --- get_gt_runnable_output ---
    def test_get_gt_runnable_output(self):
        res = get_gt_runnable_output(self.example)
        self.assertIn("SELECT", res)
        self.assertIn("<table>", res)

    # --- format_structured_output ---
    def test_format_structured_output(self):
        sel = 1
        agg = 0
        conds = {"col": [0], "op": [0], "val": ["France"]}
        headers = ["Country", "Capital"]
        result = format_structured_output(sel, agg, conds, headers=headers)
        self.assertTrue('[LBRACE]"sel": "Capital", "agg": "NULL", "conds": [LBRACE]"col": [LBRACK]"Country"[RBRACK], "op": [LBRACK]"="[RBRACK], "val": [LBRACK]"France"[RBRACK][RBRACE][RBRACE]')


if __name__ == "__main__":
    unittest.main()
