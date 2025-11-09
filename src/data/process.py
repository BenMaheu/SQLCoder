from typing import Dict, List
import json
from config import COL_SEP, SAMPLE_SEP, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH, ALLOWED_MODES, IDX2OP, AGGS_STR_ORDERED
from utils import format_string, dict2query, format_backtick


class Processor:
    def __init__(self,
                 tokenizer,
                 mode: str = "human_readable_output",
                 with_samples: bool = False,
                 column_sep: str = COL_SEP,
                 samples_sep: str = SAMPLE_SEP,
                 max_in_tokens=MAX_INPUT_LENGTH,
                 max_out_tokens=MAX_OUTPUT_LENGTH):
        self.tokenizer = tokenizer
        assert mode in ALLOWED_MODES, f"Invalid mode. Choose in {ALLOWED_MODES}."
        self.mode = mode
        self.with_samples = with_samples
        self.with_type = mode == "runnable_output"
        self.column_sep = column_sep
        self.samples_sep = samples_sep
        self.max_in_tokens = max_in_tokens
        self.max_out_tokens = max_out_tokens
        self.allowed_modes = ALLOWED_MODES

    def get_input_prompt(self, example):
        question, headers, types = example["question"], example["table"]["header"], example["table"]["types"]
        if self.with_type:
            return format_input_with_type(
                question,
                headers,
                types,
                headers_sep=self.column_sep,
                with_samples=self.with_samples,
                samples_sep=self.samples_sep
            )
        else:
            return format_input(question,
                                headers,
                                headers_sep=self.column_sep,
                                with_samples=self.with_samples,
                                samples_sep=self.samples_sep)

    def preprocess_data_row(self, example, test_mode=False):
        """Preprocesses a raw of Dataset dict for training
        {
          "question": xx,
          "table": {"header": [h1, ..., hc]},
          "sql": {
            "human_readable": xx,
            "agg": [],
            "sel": [],
            "conds": {
                "op": [],
                "val": [],
                "col": []
                }
            },
          }"""
        # Input
        example["input"] = self.get_input_prompt(example)
        model_inputs = self.tokenizer(example["input"],
                                      max_length=self.max_in_tokens,
                                      truncation=True)
        if test_mode:
            return model_inputs

        # Output
        if self.mode == "structured_output":
            example["target"] = get_gt_structured_output(example)
        elif self.mode == "human_readable_output":
            example["target"] = get_gt_human_readable_output(example)
        elif self.mode == "runnable_output":
            example["target"] = get_gt_runnable_output(example)
        else:
            raise ValueError(f"Unknown mode {self.mode}, should be in {[self.allowed_modes]}")

        labels = self.tokenizer(example["target"],
                                max_length=self.max_out_tokens,
                                truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_query(self, question: str, table_list: List[str]):
        # TODO for inference in UI
        pass


def get_gt_structured_output(example):
    sel = example["sql"]["sel"]
    agg = example["sql"]["agg"]
    conds = example["sql"]["conds"]
    headers = example["table"]["header"]
    return format_structured_output(sel, agg, conds, headers)


def get_gt_human_readable_output(example):
    return example["sql"]["human_readable"]


def get_gt_runnable_output(example):
    table_name = "<table>"
    headers = example["table"]["header"]
    sql_dict = example["sql"]
    types = example["table"]["types"]
    sql_query = dict2query(table_name, headers, sql_dict, types)
    return format_backtick(sql_query)


def format_input(question: str,
                 headers: List,
                 headers_sep=COL_SEP,
                 with_samples=False,
                 samples_sep=SAMPLE_SEP) -> str:
    """Builds the prompt"""
    text = "task: text-to-sql"
    text += "\n" + "question: " + question
    text += "\n" + "columns: " + headers_sep.join(headers)
    if with_samples:
        # TODO
        # samples: State/territory=[Australian Capital Territory, South Australia] ; Current slogan=[ACT · CELEBRATION OF A CENTURY 2013, SOUTH AUSTRALIA] ; Notes=[Slogan screenprinted on plate, No slogan on current series]
        text += "\n"
        for i in range(len(headers)):
            pass

    return format_string(text)


def format_input_with_type(question: str,
                           headers: List[str],
                           types: List[str],
                           headers_sep=" | ",
                           with_samples=False,
                           samples_sep=" ; ") -> str:
    text = "task: text-to-sql"
    text += "\n" + "question: " + question
    text += "\n" + "columns: " + headers_sep.join(headers)
    text += "\n" + "types: " + headers_sep.join(types)
    if with_samples:
        # TODO
        # samples: State/territory=[Australian Capital Territory, South Australia] ; Current slogan=[ACT · CELEBRATION OF A CENTURY 2013, SOUTH AUSTRALIA] ; Notes=[Slogan screenprinted on plate, No slogan on current series]
        text += "\n"
        for i in range(len(headers)):
            pass
    return format_string(text)


def format_structured_output(sel: int, agg: int, conds: Dict[str, List], headers: List[str]) -> str:
    """Formats the GT JSON output for training"""
    # Turn sel, agg, conds into string for better understanding from the model
    sel_str = headers[sel]
    agg_str = AGGS_STR_ORDERED[agg]
    conds_str = {"col": [], "op": [], "val": []}
    for idx in range(len(conds["col"])):
        col = headers[conds["col"][idx]]
        op = IDX2OP[conds["op"][idx]]
        val = conds["val"][idx]
        conds_str["col"].append(col)
        conds_str["op"].append(op)
        conds_str["val"].append(val)
    return format_string(json.dumps({
        "sel": sel_str,
        "agg": agg_str,
        "conds": conds_str
    }, ensure_ascii = False))
