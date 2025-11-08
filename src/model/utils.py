from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import torch
from config import NEW_TOKENS, OP2IDX


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              use_fast=False, )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # For structured output purposes : Add new special tokens
    specials = {"additional_special_tokens": list(NEW_TOKENS.values()) + ["<table>"]}
    tokenizer.add_special_tokens(specials)

    # Add tokens for operators ('<' is not in the tokenizer's vocab by default)
    tokenizer.add_tokens(list(OP2IDX.keys()))

    # Resize model's embedding
    model.resize_token_embeddings(len(tokenizer))

    # Data collator to avoid dealing with -100 / pad_token issues
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    return model, tokenizer, data_collator


def load_model_from_checkpoint(model_checkpoint: str):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    # Load training arguments
    training_args = torch.load(model_checkpoint + "/training_args.bin", weights_only=False)

    return model, tokenizer, training_args
