"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.

Sample code that uses start_flask_server to load a set of T5-large models.

When first checking out this repository, the model checkpoints and outputs will
not be available and throw an error upon visiting the Behaviors or Layers views.
To generate these models, run `python make_t5_models.py` (may take several hours
on GPU).
"""

import os
import torch
import numpy as np
import h5py

from interactive_compression.server import start_flask_server
from interactive_compression.runners import LocalRunner
from interactive_compression.inspectors.pytorch import PyTorchModelInspector
from interactive_compression.inspectors.base import Prediction
from interactive_compression.monitors import LocalModelMonitor

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments
from trainer_seq2seq_qa import QuestionAnsweringSeq2SeqTrainer

import t5_utils

device = (
    "cuda"
    if torch.cuda.is_available()
    else (
        "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
)
batch_size = 8

model_path = os.path.join(os.path.dirname(__file__), "models")

tokenizer = AutoTokenizer.from_pretrained("t5-large")

monitor = LocalModelMonitor(model_path, outputs_type="json")


class T5ModelInspector(PyTorchModelInspector):
    ACTIVATION_EXCLUDE_PATTERNS = ["^cls", "dropout$"]

    def get_model(self, model_id, for_inference=False):
        print("Loading existing fine-tuned model from", model_path)
        self.model_id = model_id
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            "t5-large").to('cpu')
        base_model.load_state_dict(
            monitor.load_model_checkpoint(model_id, map_location='cpu'))

        self.args = Seq2SeqTrainingArguments(
            "tmp_model_dir",
            evaluation_strategy="no",
            save_strategy="epoch",
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

        self.eval_examples, self.eval_dataset, self.data_collator = t5_utils.load_data(
            tokenizer)

        # Initialize our Trainer
        self.trainer = QuestionAnsweringSeq2SeqTrainer(
            model=base_model,
            args=self.args,
            train_dataset=None,  # if training_args.do_train else None,
            eval_dataset=self.eval_dataset,  # if training_args.do_eval else None,
            eval_examples=self.eval_examples,
            tokenizer=tokenizer,
            data_collator=self.data_collator,
            compute_metrics=None,  # if training_args.predict_with_generate else None,
            post_process_function=None,
        )

        if for_inference:
            return base_model.to(device)

        return base_model

    def get_eval_dataloader(self):
        return self.trainer.get_eval_dataloader()

    def inference_batch(self, model, batch, batch_index):
        outputs1 = model.generate(input_ids=batch["input_ids"].to(
            device), max_length=1, num_beams=1)

    def predictions(self, model_id):
        preds = monitor.get_model_outputs(model_id)

        predictions = [
            Prediction(
                i,
                pred["label"],
                pred["pred"],
                [],
            ).to_dict()
            for i, pred in enumerate(preds)
        ]

        tensor_filepath = os.path.join(self.output_path, "tensors.h5")
        tensor_file = h5py.File(tensor_filepath, "w")

        tensor_file.create_dataset(
            "f1_scores", data=np.array([pred["f1"] for pred in preds]))
        tensor_file.create_dataset("exact_match_scores", data=np.array(
            [pred["exact_match"] for pred in preds]))

        tensor_file.close()

        return {
            "predictions": predictions,
            "comparisons": {
                "F1": {
                    "comparison": "difference",
                    "vectors": {
                        "format": "hdf5",
                        "path": tensor_filepath,
                        "dataset_name": "f1_scores",
                    },
                },
                "Exact Match Score": {
                    "comparison": "difference",
                    "vectors": {
                        "format": "hdf5",
                        "path": tensor_filepath,
                        "dataset_name": "exact_match_scores",
                    },
                },
            },
        }

    def instances(self, input_ids):
        if not hasattr(self, "eval_examples"):
            self.eval_examples, self.eval_dataset, self.data_collator = t5_utils.load_data(
                tokenizer)
        selected_dataset = self.eval_examples.select(input_ids)
        return {
            "type": "text",
            "instances": {
                id: {
                    "text": f"{example[t5_utils.question_column]} \n\n{example[t5_utils.context_column]}",
                }
                for id, example in zip(input_ids, selected_dataset)
            },
        }


if __name__ == "__main__":
    start_flask_server(
        monitor,
        LocalRunner(
            T5ModelInspector,
            os.path.join(os.path.dirname(__file__), "task_outputs"),
            max_workers=1,
        ),
        debug=False
    )
