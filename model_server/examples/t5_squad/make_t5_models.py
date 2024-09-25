"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.

Generates T5-large models for SQuAD question-answering. This may take several hours
to run on a GPU.
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments
import t5_utils
import evaluate
from functools import partial
from shutil import rmtree
from typing import List, Optional, Tuple
from trainer_seq2seq_qa import QuestionAnsweringSeq2SeqTrainer

from interactive_compression.monitors import LocalModelMonitor
import interactive_compression.monitors.pytorch_metric_utils as metric_utils
from interactive_compression.utils import empty_device_cache

model_path = os.path.join(os.path.dirname(__file__), "models")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

tokenizer = AutoTokenizer.from_pretrained("t5-large")

train_dataset, eval_examples, eval_dataset, data_collator = t5_utils.load_data(
    tokenizer, make_train_data=True)

# evaluate.load("squad_v2" if data_args.version_2_with_negative else "squad")
metric = evaluate.load("squad")


def make_model(device='cpu'):
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-large").to(device)
    return model


def generation_run_fn(model, batch):
    def runner():
        return model.generate(**batch, max_length=16, num_beams=1)
    return runner


def compute_metrics(model, dataloader, labels, outputs, checkpoint):
    results = {}
    squad_scores = metric.compute(predictions=outputs, references=labels)
    results['ExactMatch'] = squad_scores['exact_match']
    results['F1'] = squad_scores['f1']

    input = next(iter(dataloader))
    print("Computing latency")
    results['Batch Latency'] = metric_utils.compute_latency(
        model,
        input,
        run_fn=generation_run_fn,
        device=device,
    )

    print("Computing model size")
    results['Model Size'] = metric_utils.compute_model_size(model)

    return results


def evaluate_model(model):
    model = model.to(device)

    max_length = 16
    num_beams = None

    args = Seq2SeqTrainingArguments(
        "tmp-model-eval",
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=3e-5,
        num_train_epochs=2,
        weight_decay=0.01,
        per_device_eval_batch_size=8,
        per_device_train_batch_size=8,
        # remove_unused_columns=False,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
        # eval_accumulation_steps=128,
    )

    # Initialize our Trainer
    trainer = QuestionAnsweringSeq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=None,  # if training_args.do_train else None,
        eval_dataset=eval_dataset,  # if training_args.do_eval else None,
        eval_examples=eval_examples,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,  # if training_args.predict_with_generate else None,
        post_process_function=partial(
            t5_utils.post_processing_function, tokenizer),
    )

    dataloader = trainer.get_eval_dataloader()

    gen_kwargs = {"max_length": max_length, "num_beams": None}
    if gen_kwargs.get("num_beams") is None and trainer.args.generation_num_beams is not None:
        gen_kwargs["num_beams"] = trainer.args.generation_num_beams
    trainer._gen_kwargs = gen_kwargs

    outputs = trainer.evaluation_loop(dataloader, description="Evaluation")
    outputs = t5_utils.post_processing_function(
        tokenizer, eval_examples, eval_dataset, outputs)

    print(outputs)
    # Compute F1 per-instance
    instance_outputs = []
    for i, (pred, label) in enumerate(zip(outputs.predictions, outputs.label_ids)):
        instance_metrics = metric.compute(
            predictions=[pred], references=[label])
        instance_outputs.append({'id': pred['id'], 'label': ', '.join(
            label['answers']['text']), 'pred': pred['prediction_text'], **instance_metrics})

    metrics = compute_metrics(
        model, dataloader, outputs.label_ids, outputs.predictions, None)
    print(metrics)
    return metrics, instance_outputs


def create_and_evaluate_base_model(monitor):
    print("Base")
    model = make_model(device='cpu')
    metrics, outputs = evaluate_model(model)
    monitor.save_model(
        "base", metrics, checkpoint=model.state_dict(), outputs=outputs)


def create_and_evaluate_pruned_model(monitor, base, sparsity, layer_patterns=None, layer_pattern_id=None):
    print("Prune", sparsity)
    model = make_model(device='cpu')

    with torch.no_grad():
        model.load_state_dict(
            monitor.load_model_checkpoint(base, map_location='cpu'))

    layers_to_prune, module_names = t5_utils.get_prunable_layers(model)
    if layer_patterns is not None:
        layers_to_prune, module_names = zip(*((layer, name)
                                              for layer, name in zip(layers_to_prune, module_names)
                                              if any(re.search(pattern, name) is not None for pattern in layer_patterns)))
    print("Layers to prune:", module_names)
    t5_utils.make_prune_fn(layers_to_prune)(model, sparsity)

    if base != "base":
        model_id = f"{base}_pruning{sparsity:.1f}"
    else:
        model_id = f"pruning{sparsity:.1f}"
    if layer_pattern_id is not None:
        model_id += "-" + layer_pattern_id
    metrics, outputs = evaluate_model(model)
    monitor.save_model(model_id, metrics, base_id=base, operation={
        "name": "Pruning" if layer_patterns is None else "Selective Pruning",
        "parameters": {"sparsity": sparsity} if layer_patterns is None else {
            "sparsity": sparsity,
            "layers": layer_pattern_id
        }
    }, checkpoint=model.state_dict(), outputs=outputs)


def create_and_evaluate_restored_model(monitor, base_id, restore_id, layer_patterns):
    print(base_id, "Restore", layer_patterns)
    model = make_model(device='cpu')

    with torch.no_grad():

        state_dict = monitor.load_model_checkpoint(base_id, map_location='cpu')

        print({k for k in state_dict.keys()
               if any(re.search(pattern, k) is not None for pattern in layer_patterns)})
        model.load_state_dict({k: (v
                                   if not any(re.search(pattern, k) is not None for pattern in layer_patterns)
                                   else model.get_parameter(k))
                               for k, v in state_dict.items()})

    model_id = base_id + "_" + f"restore{restore_id}"
    del state_dict
    metrics, outputs = evaluate_model(model)
    monitor.save_model(model_id, metrics, base_id=base_id, operation={"name": "Restore", "parameters": {
                       "layers": restore_id}}, checkpoint=model.state_dict(), outputs=outputs)


def create_and_evaluate_retrained_model(monitor, base_id):
    print(base_id, "Retrain")
    model = make_model(device='cpu')

    with torch.no_grad():
        model.load_state_dict(monitor.load_model_checkpoint(
            base_id, map_location='cpu'))

    for p in model.parameters():
        p.requires_grad = False
    for name, module in model.named_modules():
        if re.search(r"layer_?norm", name) is not None:
            for p in module.parameters():
                if name == "encoder.block.0.layer.0.layer_norm":
                    print(p)
                p.requires_grad = True

    model = model.to(device)

    model_id = base_id + "_" + f"retrain"
    checkpoint_dir = os.path.join(model_path, model_id + "_checkpoints")

    args = Seq2SeqTrainingArguments(
        checkpoint_dir,
        evaluation_strategy="no",
        save_strategy="no",
        learning_rate=1e-3,
        num_train_epochs=0.1,
        weight_decay=0.01,
        per_device_eval_batch_size=8,
        per_device_train_batch_size=8,
        # remove_unused_columns=False,
        predict_with_generate=True,
        fp16=False,
        push_to_hub=False,
        # eval_accumulation_steps=128,
    )

    # Initialize our Trainer
    trainer = QuestionAnsweringSeq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,  # if training_args.do_train else None,
        eval_dataset=eval_dataset,  # if training_args.do_eval else None,
        eval_examples=eval_examples,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,  # if training_args.predict_with_generate else None,
        post_process_function=partial(
            t5_utils.post_processing_function, tokenizer),
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    print(train_result, metrics)
    for p in model.parameters():
        p.requires_grad = False
    for name, module in model.named_modules():
        if name == "encoder.block.0.layer.0.layer_norm":
            for p in module.parameters():
                print(p)

    if os.path.exists(checkpoint_dir):
        rmtree(checkpoint_dir)

    metrics, outputs = evaluate_model(model)
    monitor.save_model(model_id, metrics, base_id=base_id, operation={"name": "Retrain", "parameters": {
                       "layers": "layernorms"}}, checkpoint=model.state_dict(), outputs=outputs)


if __name__ == '__main__':

    monitor = LocalModelMonitor(model_path, outputs_type="json")
    monitor.set_operation("Pruning", sparsity={
                          "type": "continuous", "min": 0, "max": 1})
    monitor.set_operation("Selective Pruning", sparsity={"type": "continuous", "min": 0, "max": 1}, layers={
                          "type": "ordinal", "options": ["attention", "feedforward"]})
    monitor.set_operation("Restore", layers={"type": "ordinal", "options": [
                          "layernorms", "query", "decoder"]})

    monitor.set_metric("ExactMatch", False, format=".1~", min=0, max=100)
    monitor.set_metric("F1", True, format=".1~", min=0, max=100)
    monitor.set_metric("Batch Latency", True, format=".2~", unit="s", min=0)
    monitor.set_metric("Model Size", True, format=".3~s", unit="B", min=0)

    empty_device_cache()

    if not monitor.has_model_checkpoint("base"):
        create_and_evaluate_base_model(monitor)
        empty_device_cache()

    # Plain magnitude pruning
    for sparsity in [0.1, 0.3, 0.5, 0.7, 0.9]:
        if not monitor.has_model_checkpoint(f"pruning{sparsity:.1f}"):
            create_and_evaluate_pruned_model(monitor, "base", sparsity)
        empty_device_cache()

    # Restoring particular layers
    for sparsity in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for restore_id, patterns in [("layernorms", [r"layer_?norm"]), ("query", [r"\.q(\.|$)"]), ("decoder", [r"^decoder"])]:
            id = f"pruning{sparsity:.1f}_restore{restore_id}"
            if not monitor.has_model_checkpoint(id):
                create_and_evaluate_restored_model(
                    monitor, f"pruning{sparsity:.1f}", restore_id, patterns)
                empty_device_cache()

            id = f"pruning{sparsity:.1f}_retrain"
            if not monitor.has_model_checkpoint(id):
                create_and_evaluate_retrained_model(
                    monitor, f"pruning{sparsity:.1f}")
                empty_device_cache()

            id = f"pruning{sparsity:.1f}_restore{restore_id}_retrain"
            if not monitor.has_model_checkpoint(id):
                create_and_evaluate_retrained_model(
                    monitor, f"pruning{sparsity:.1f}_restore{restore_id}")
                empty_device_cache()

    # More pruning on either attention or FFN layers
    for sparsity in [0.4, 0.5, 0.6]:
        for pattern_id, patterns in [("attention", [r"Attention"]), ("feedforward", [r"Dense"])]:
            id = f"pruning0.3_restorelayernorms_pruning{sparsity:.1f}-{pattern_id}"
            if not monitor.has_model_checkpoint(id):
                create_and_evaluate_pruned_model(
                    monitor,
                    f"pruning0.3_restorelayernorms",
                    sparsity,
                    layer_patterns=patterns,
                    layer_pattern_id=pattern_id)
                empty_device_cache()

    # Even more pruning on late attention blocks
    for sparsity in [0.4, 0.5]:
        # ("feedforward", [r"Dense"]),
        for pattern_id, patterns in [("lateattention", [r"\.block\.1\d\..*Attention"])]:
            base_id = f"pruning0.3_restorelayernorms_pruning{sparsity:.1f}-attention"
            id = base_id + f"_pruning{sparsity + 0.2:.1f}-lateattention"
            if not monitor.has_model_checkpoint(id):
                create_and_evaluate_pruned_model(
                    monitor,
                    base_id,
                    sparsity + 0.2,
                    layer_patterns=patterns,
                    layer_pattern_id=pattern_id)
                empty_device_cache()
