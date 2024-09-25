# Compress and Compare Image Compression Experiments

Compress and Compare comes preloaded with example compression experiments on CIFAR-10, CelebA, and ImageNet. This folder contains the code used to generate those models and analyze thier performance.

## Usage
Usage examples are shown in `usage_example.ipynb`.

### Training and Compressing
To train and compress models on each of the datasets, use `run_compression_experiment` in `experiments.py`. 

For example, to train a CIFAR-10 ResNet18 and apply 50% magnitude pruning with retraining:
```
experiments.run_compression_experiment(
    dataset_name='cifar10',                                  # use CIFAR-10 dataset
    model_fn=resnet.resnet20,                                # use a ResNet20
    output_dir='./results/',                                 # save outputs to ./results/
    checkpoint_name=None,                                    # do not load from chackpoint
    compression_fns=[compression.magnitudepruning],          # apply one round of magnitude pruning
    compression_amounts=[0.5],                               # prune 50% of the weights
    train_num_epochs=200,                                    # train for 200 epochs
    retrain_num_epochs=[50],                                 # after compression train for 50 epochs
    recalibrate=False,                                       # do not recalibrate batch norm layers
    device='cuda',                                           # run on GPU
    model_fn_kwargs={'num_classes': 10, 'quantized': False}, # ResNet20 has 10 output classes and no quantization
    verbose=True,                                            # print outputs during experiment
    iterative_compression=False,                             # no iterative compression
    allow_weight_updates=False                               # pruned weights can not be updated
)
```

You can also call directly from the command line:
```
experiments.py -n cifar10 -m resnet.resnet20 -o ./results/ -f compression.magnitudepruning -a 0.5 -e 200 -r 50 -d cuda -k {'num_classes':10,'quantized':False} -v
```

### Compression Performance Metrics
To compute model outputs and performance metrics used to generate the `models.json` files that support Compress and Compare (e.g., `model_server/examples/imagenet_image_classification/imagenet_image_classification_models.json`), use `analyze_performance` in `compression_performance.py`.

For example to compute metrics on all ImageNet models saved in `'./imagenet_results/'`:
```
analyze_performance(
    model_dir='./imagenet_results/',
    outputs_dir='./imagenet_results/outputs/',
    dataset='imagenet'
)
```

You can also call directly from the command line:
```
compression_performance.py --model_dir ./imagenet_results/ --outputs_dir ./imagenet_results/outputs/ --dataset imagenet
```
