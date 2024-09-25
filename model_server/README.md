# Interactive Compression

This directory contains the Python package that connects to your ML code and serves models and data to the Compress and Compare interface.

To display and inspect a new set of models, you will need to write a script that calls the `start_flask_server` function. This function takes the following arguments:

- `model_info`: a model monitor object, or a dictionary describing the available metrics, operations, and models. The dictionary structure is documented in `server.py`.
- `task_runner`: an instance of a task runner (such as `local_runner.LocalRunner`) that can run arbitrary task commands. `LocalRunner` uses a background scheduling mechanism to run model inspection tasks in parallel on the local machine.

As described in the top-level readme, the examples in the `model_server/examples` directory show several ways that the model server can be constructed.

## Model Monitors

The first parameter to the model server can be a **model monitor** object, which is typically an instance of a class such as `monitors.LocalModelMonitor` (more monitor variants TBD). A model monitor maintains a set of models, metrics about them, and the ability to load weights and (optionally) instance-level outputs for each model. 

The easiest way to create a model monitor for a set of experiments that run synchronously on a single device is to initialize a `LocalModelMonitor` during model experimentation, point it to a directory where you want model information to be stored, and add models to it as you generate them. You'll need to call the following methods at some point during the experiment's execution:

* `set_operation` - define an operation and its parameters. For instance, `monitor.set_operation('Pruning', sparsity={'type': 'continuous', 'min': 0, 'max': 1})` defines a pruning operation with a single parameter named "sparsity". Operations must be defined before they are used in model metadata.
* `set_metric` - define a performance metric. For instance, `monitor.set_metric('Latency', primary=True, format='.3~s', unit='s', min=0)` defines a latency metric with a d3 format string defining how values should be displayed in the interface. Metrics must be defined before they are used when saving a model.
* `save_model` - add a new model, and optionally save its weights and outputs. All models have an ID and a dictionary of metrics, and they can optionally have a `base_id` and an `operation` to denote that they are derived from the model with ID `base_id`.

As you call these methods on the model monitor, they will be saved automatically to the directory you specified in the object's initialization. Then, in the script for the model server, you can simply reinstantiate a `LocalModelMonitor` pointed to the same directory, and pass it directly to the `start_flask_server` function.

## Model Inspectors

When you create a task runner, you will need to define a **model inspector** class that provides the required information about your dataset and models. We have implemented a basic model inspector class, `inspectors.pytorch.PyTorchModelInspector`, which you can customize by creating subclasses. The following methods of the PyTorch model inspector _must_ be overridden:

- `get_model` - load the model from disk
- `get_eval_dataloader` - load a dataloader to iterate on to produce evaluation results (must be deterministic)
- `inference_batch` - run a forward pass on the given batch of instances, and return predictions and associated tensors
- `comparison_metrics` - define what comparisons should be done given the outputs of `inference_batch`
- `instances` - retrieve instances with the given IDs

You may also override `iter_modules` to define a custom set of modules to display in the Layers view.

In summary, an implementation of starting the compression visualization tool with a new model and dataset would look like this:

```python
class MyModelInspector(PyTorchModelInspector):
    def get_model(self, model_id, for_inference=False):
        # load the model

    def get_eval_dataloader(self):
        # load the dataloader for evaluation

    def inference_batch(self, model, batch, batch_index):
        outputs, labels = model(**batch)

        predictions = [
            Prediction(
                batch_index * batch_size + i,
                # ... ground truth label string, model prediction, and ground-truth classes
            )
            # ...
        ]
        # save other outputs as tensors
        outputs = torch.cat(outputs, 0)
        return predictions, {
            "scores": scores,
            "logits": outputs.cpu().numpy().astype(np.float16),
            # ...
        }

    def comparison_metrics(self):
        return {
            "Confidence on True Label": ("difference", "true_probs"),
            # ... other metrics
        }

    def instances(self, input_ids):
        # get instances with the given ids
        return {
            "type": "text", # or "image"
            "instances": {
                id: {
                    # ...
                }
                for id in input_ids
            },
        }

if __name__ == "__main__":
    start_flask_server(
        model_info_dictionary,
        LocalRunner(
            MyModelInspector(),
            os.path.join(os.path.dirname(__file__), "task_outputs"),
            max_workers=1
        ),
    )
```

If you are creating a new model inspector from scratch (e.g. not PyTorch-based), the `inspectors.base.ModelInspectorBase` class offers a starting point with method signatures to implement these methods on your own. See the documentation of `ModelInspectorBase` for how the inputs and outputs to those method signatures are formatted.

## Internal Task Runner Details

These details explain how the model inspector is used under the hood, specifically by the task runner.

The `model_inspector` argument to the task runner is called with two parameters: a dictionary of arguments that is guaranteed to contain the `command` key, and an output directory path. The responsibility of this function is to write a file called `output.json` to the output directory that contains the required information requested by the command. The output file should be a JSON file containing a `status` key (which can be `running`, `complete`, or `error`). When the status is `running`, the file can contain a `message` string and a `progress` number indicating the current activity. When the status is `complete`, it should contain a `result` key that matches the structure given the command:

### Command: `sparsity`

Arguments: `model_id` (str)

```python
{
    "module.submodule.layer": {
        "module_name": str,
        "parameter_name": str,
        "num_zeros": int,
        "num_parameters": int,
        "sparsity_mask": {
            "path": str,
            "format": "hdf5", # currently only hdf5 is supported
            "dataset_name": str
        }
    },
    # ...
}
```

The sparsity mask file should be an HDF5 file containing multiple datasets, each of which corresponds to a binary pruning mask (1 = unpruned, 0 = pruned). The file can be saved to the output directory, though it is not required as long as the path is accessible by the server.

The hierarchy of modules will be determined by the module names, such that `A.B` is interpreted as a child of module `A`.

### Command: `weights`

Arguments: `model_id` (str)

```python
{
    "module.submodule.layer": {
        "module_name": str,
        "parameter_name": str,
        "weights": {
            "path": str,
            "format": "hdf5", # currently only hdf5 is supported
            "dataset_name": str
        }
    },
    # ...
}
```

The sparsity mask file should be an HDF5 file containing multiple datasets, each of which corresponds to a tensor containing weights. The file can be saved to the output directory, though it is not required as long as the path is accessible by the server.

The hierarchy of modules will be determined by the module names, such that `A.B` is interpreted as a child of module `A`.

### Command: `predictions`

Arguments: `model_id` (str)

```python
{
    "predictions": [
        {
            "id": int,
            "label": str,
            "pred": str,
            "classes": List[str],
        },
        #...
    ],
    "comparisons": {
        "Comparison Name": {
            "comparison": "difference" | "abs_difference" | "difference_argmax" | "abs_difference_argmax" | "ratio" | "ratio_argmax" | "mse" | "kl_divergence",
            "vectors": {
                "format": "hdf5",
                "path": str,
                "dataset_name": str
            }
        },
        # ...
    }
}
```

The `predictions` list simply lists all instances that were evaluated, and provides their ground-truth label and the model prediction for that instance. These values are displayed directly in the interface and can have any string representation.

The `comparison` key determines how the model server will compare the given model outputs against the selected base model. If the `comparison` is `difference` or `ratio`, each row should only contain 1 value (otherwise the values will be averaged together).

### Command: `instances`

Arguments: `ids` (List of strings)

```python
{
    "type": "text",
    "instances": {
        id: {
            "text": str,
        },
        #...
    }
}
```

OR:

```python
{
    "type": "image",
    "instances": {
        id: {
            "image": str # base-64 encoded string,
            "image_format": str # e.g. 'image/png'
        },
        #...
    }
}
```

Each key of the returned `instances` dictionary should be one of the IDs passed in the input `ids` arg.
