import evaluate
import random

from utils import extract_json, strip_specials, unformat_string

rouge = evaluate.load("rouge")


def check_lf(pred, label):
    """Returns 0 if non logic form compliant 1 otherwise"""
    if pred["sel"] != label["sel"]:
        return 0
    if pred["agg"] != label["agg"]:
        return 0

    pred_conds = pred["conds"]
    label_conds = label["conds"]

    # WARNING : Here we want some order-insensitive comparison on conditions
    pred_conds_set = set(zip(pred_conds["col"], pred_conds["op"], pred_conds["val"]))
    label_conds_set = set(zip(label_conds["col"], label_conds["op"], label_conds["val"]))
    if pred_conds_set != label_conds_set:
        return 0

    return 1


def logic_form_accuracy(preds, labels):
    correct = 0
    incorrect_parse = 0

    for pred, label in zip(preds, labels):
        try:
            pred = extract_json(pred)
            label = extract_json(label)
        except:
            incorrect_parse += 1
            continue

        correct += check_lf(pred, label)
    return {"lfa": correct / len(preds), "incorrect_json_ratio": incorrect_parse / len(preds)}


def slot_level_accuracy(preds, labels):
    total_conds, correct_sel, correct_agg, correct_cond_col, correct_cond_op, correct_cond_value = 0, 0, 0, 0, 0, 0

    for pred, label in zip(preds, labels):
        try:
            pred = extract_json(pred)
            label = extract_json(label)
        except:
            continue

        if pred["sel"] == label["sel"]:
            correct_sel += 1
        if pred["agg"] == label["agg"]:
            correct_agg += 1

        # Condition-level accuracy (order-insensitive)
        pred_conds = set(zip(pred["conds"]["col"], pred["conds"]["op"], pred["conds"]["val"]))
        label_conds = set(zip(label["conds"]["col"], label["conds"]["op"], label["conds"]["val"]))

        total_conds += len(label_conds)
        correct_conds = pred_conds & label_conds

        correct_cond_col += len({c[0] for c in correct_conds})
        correct_cond_op += len({c[1].strip() for c in correct_conds})
        correct_cond_value += len({c[2] for c in correct_conds})

    denom = total_conds + 1e-12
    return {
        "sel_acc": correct_sel / (len(preds) + 1e-12),
        "agg_acc": correct_agg / (len(preds) + 1e-12),
        "cond_col_acc": correct_cond_col / denom,
        "cond_op_acc": correct_cond_op / denom,
        "cond_value_acc": correct_cond_value / denom,
    }


# For structured output
def compute_metrics(pred, tokenizer):
    preds, labels = pred.predictions, pred.label_ids

    # Remove all unnecessary tokens (pad them)
    preds[preds == -100] = tokenizer.pad_token_id
    preds_str = tokenizer.batch_decode(preds, skip_special_tokens=False)

    labels[labels == -100] = tokenizer.pad_token_id
    labels_str = tokenizer.batch_decode(labels, skip_special_tokens=False)

    # Print for process tracking
    sample_preds_indices = random.sample(range(len(preds_str)), 3)
    for i in sample_preds_indices:
        print("Raw pred : ", preds_str[i])
        print("Raw labels : ", labels_str[i])
        print("Clean pred : ", unformat_string(strip_specials(preds_str[i])))
        print("Clean labels : ", unformat_string(strip_specials(labels_str[i])))
        print("-" * 50 + "\n\n")

    # Compute metrics
    lfa, incorrect_json_ratio, sla, sel_acc, agg_acc, cond_col_acc, cond_op_acc, cond_value_acc, total_conds = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for pred, label in zip(preds_str, labels_str):
        try:
            pred = extract_json(pred)
            label = extract_json(label)
        except:
            continue

        # Logic form accuracy
        lfa += check_lf(pred, label)

        # Slot level accuracy
        if pred["sel"] == label["sel"]:
            sel_acc += 1
        if pred["agg"] == label["agg"]:
            agg_acc += 1

        total_conds += len(label["conds"]["col"])
        for i in range(min(len(pred["conds"]["col"]), len(label["conds"]["col"]))):
            try:
                if pred["conds"]["col"][i] == label["conds"]["col"][i]:
                    cond_col_acc += 1
                if pred["conds"]["op"][i] == label["conds"]["op"][i]:
                    cond_op_acc += 1
                if pred["conds"]["val"][i] == label["conds"]["val"][i]:
                    cond_value_acc += 1
            except:
                continue

    total_conds += 1e-12  # Avoid division by 0
    print("\n\n")

    return {
        # Macro
        "logic_form_accuracy": lfa / len(preds_str),
        "incorrect_json_ratio": incorrect_json_ratio / len(preds_str),

        # Micro
        "sel_accuracy": sel_acc / len(preds_str),
        "agg_accuracy": agg_acc / len(preds_str),
        "cond_col_accuracy": cond_col_acc / total_conds,
        "cond_op_accuracy": cond_op_acc / total_conds,
        "cond_value_accuracy": cond_value_acc / total_conds,
    }


def compute_human_readable_metrics(pred, tokenizer):
    preds, labels = pred.predictions, pred.label_ids

    # Remove all unnecessary tokens (pad them)
    preds[preds == -100] = tokenizer.pad_token_id
    preds_str = tokenizer.batch_decode(preds, skip_special_tokens=False)

    labels[labels == -100] = tokenizer.pad_token_id
    labels_str = tokenizer.batch_decode(labels, skip_special_tokens=False)

    rouge_output = rouge.compute(predictions=preds_str,
                                 references=labels_str)

    # Print for process tracking
    sample_preds_indices = random.sample(range(len(preds_str)), 3)
    for i in sample_preds_indices:
        print("Raw pred : ", preds_str[i])
        print("Raw labels : ", labels_str[i])
        print("Clean pred :   ", unformat_string(strip_specials(preds_str[i])))
        print("Clean labels : ", unformat_string(strip_specials(labels_str[i])))
        print("-" * 50 + "\n\n")

    exact_match_accuracy, total = 0, 0
    for pred, label in zip(preds_str, labels_str):
        total += 1
        clean_pred = unformat_string(strip_specials(pred))
        clean_label = unformat_string(strip_specials(label))
        if clean_pred.strip() == clean_label.strip():
            exact_match_accuracy += 1

    rouge_output["exact_match_accuracy"] = exact_match_accuracy / (total + 1e-12)
    return rouge_output
