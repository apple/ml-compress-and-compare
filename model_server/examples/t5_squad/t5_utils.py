"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import torch
import gc
from torch.nn.utils import prune
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, DefaultDataCollator, DataCollatorForSeq2Seq, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, AutoModelForCausalLM, TrainingArguments, Trainer, Seq2SeqTrainingArguments
import datasets
from typing import List, Optional, Tuple
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, get_last_checkpoint
from trainer_seq2seq_qa import QuestionAnsweringSeq2SeqTrainer
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import (
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
)
from functools import partial
from interactive_compression.torch import global_unstructured_patch, calculate_sparsity


def get_prunable_layers(model):
    prunable_modules = []
    module_names = []
    for name, module in model.named_modules():
        if name.split(".")[0] not in ("encoder", "decoder"):
            continue
        for param_name in ("weight",):
            if hasattr(module, param_name):
                if getattr(module, param_name).requires_grad:
                    prunable_modules.append((module, param_name))
                    module_names.append(name)
    return prunable_modules, module_names


prune_batch_size = 1


def prune_global_l1(layers_to_prune, device="cpu"):
    def prune_fn(model, sparsity):
        if sparsity > 0.0:
            _, global_sparsity_before = calculate_sparsity(layers_to_prune)
            global_unstructured_patch(
                layers_to_prune, prune.L1Unstructured, amount=sparsity
            )
            _, global_sparsity_after = calculate_sparsity(layers_to_prune)
            return global_sparsity_after
        return sparsity

    return prune_fn


def make_prune_fn(layers_to_prune):
    def prune_fn(model, sparsity):
        print("Pruning at sparsity", sparsity)
        _, global_sparsity_before = calculate_sparsity(layers_to_prune)
        for i in tqdm(range(0, len(layers_to_prune) // prune_batch_size)):
            batch_layers = layers_to_prune[i *
                                           prune_batch_size:(i + 1) * prune_batch_size]
            prune_global_l1(batch_layers)(model, sparsity)
            for layer in batch_layers:
                prune.remove(*layer)
            gc.collect()
        _, global_sparsity_after = calculate_sparsity(layers_to_prune)
        print(
            f"True sparsity: {global_sparsity_before} -> {global_sparsity_after}")
    return prune_fn


# SQUAD DATA CODE

squad = load_dataset("squad")

raw_datasets = squad

question_column = "question"
context_column = "context"
answer_column = "answers"
column_names = squad["train"].column_names

# Temporarily set max_answer_length for training.
max_answer_length = 30  # data_args.max_answer_length
padding = "max_length"  # if data_args.pad_to_max_length else False

max_seq_length = 384  # min(384, tokenizer.model_max_length)


def preprocess_squad_batch(
    examples,
    question_column: str,
    context_column: str,
    answer_column: str,
) -> Tuple[List[str], List[str]]:
    questions = examples[question_column]
    contexts = examples[context_column]
    answers = examples[answer_column]

    def generate_input(_question, _context):
        return " ".join(["question:", _question.lstrip(), "context:", _context.lstrip()])

    inputs = [generate_input(question, context)
              for question, context in zip(questions, contexts)]
    targets = [answer["text"][0] if len(
        answer["text"]) > 0 else "" for answer in answers]
    return inputs, targets


def preprocess_function(tokenizer, examples):
    inputs, targets = preprocess_squad_batch(
        examples, question_column, context_column, answer_column)

    model_inputs = tokenizer(
        inputs, max_length=max_seq_length, padding=padding, truncation=True)
    # Tokenize targets with text_target=...
    labels = tokenizer(
        text_target=targets, max_length=max_answer_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and True:  # data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Validation preprocessing


def preprocess_validation_function(tokenizer, examples):
    inputs, targets = preprocess_squad_batch(
        examples, question_column, context_column, answer_column)

    model_inputs = tokenizer(
        inputs,
        max_length=max_seq_length,
        padding=padding,
        truncation=True,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=targets, max_length=max_answer_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and True:  # data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = model_inputs.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    model_inputs["example_id"] = []
    # Augment the overflowing tokens to the labels
    labels_out = []

    for i in range(len(model_inputs["input_ids"])):
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        model_inputs["example_id"].append(examples["id"][sample_index])
        labels_out.append(labels["input_ids"][sample_index])

    model_inputs["labels"] = labels_out
    return model_inputs


def load_data(tokenizer, make_train_data=False):
    print("Loading dataset...")

    eval_examples = raw_datasets["validation"]
    print("Got examples, loading dataset")

    # Validation Feature Creation
    eval_dataset = eval_examples.map(
        partial(preprocess_validation_function, tokenizer),
        batched=True,
        num_proc=None,  # data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,  # not data_args.overwrite_cache,
        desc="Running tokenizer on validation dataset",
    )

    # Data collator
    # if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        # model=model,
        max_length=384,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8  # if training_args.fp16 else None,
    )

    results = (eval_examples, eval_dataset, data_collator)

    if make_train_data:
        train_dataset = raw_datasets["train"]
        # Create train feature from dataset
        train_dataset = train_dataset.map(
            partial(preprocess_function, tokenizer),
            batched=True,
            num_proc=4,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on train dataset",
        )
        results = (train_dataset, *results)

    return results

# Post-processing:


def post_processing_function(
    tokenizer, examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, stage="eval"
):
    # Decode the predicted tokens.
    print("Post process")
    preds = outputs.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    feature_per_example = {
        example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
    predictions = {}
    # Let's loop over all the examples!
    for example_index, example in enumerate(examples):
        # This is the index of the feature associated to the current example.
        feature_index = feature_per_example[example_index]
        predictions[example["id"]] = decoded_preds[feature_index]

    # Format the result to the format the metric expects.
    # if data_args.version_2_with_negative:
    #     formatted_predictions = [
    #         {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
    #     ]
    # else:
    formatted_predictions = [{"id": k, "prediction_text": v}
                             for k, v in predictions.items()]

    references = [{"id": ex["id"], "answers": ex[answer_column]}
                  for ex in examples]
    print("Done post processing")
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)
