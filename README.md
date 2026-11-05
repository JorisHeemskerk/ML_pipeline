# ML_pipeline
This GitHub repo is a personallised implementation of ML trainer platforms like ML Flow and Pytorch Lightning. It is more suited to my personal needs.

---

# Training

The training engine of the pipeline. The root `trainer.py` reads the
`trainer` section of a job config and drives this package, so a new
project normally only needs a config, and modules for whatever is new.

## Layout

| File | Purpose |
|------|---------|
| `trainer.py` | `TrainerConfig` and `Trainer` (engine): split training, k-fold cross validation, evaluation. |
| `callbacks.py` | `Callback` base class, `CALLBACKS` registry, `EarlyStoppingCallback`, `PruningCallback`, `CheckpointCallback`. |
| `feature_resampling.py` | `DeadFeatureResamplingCallback`: revives dead features of a sparse autoencoder. |
| `early_stopper.py` | `EarlyStopper`, used by `EarlyStoppingCallback`. |
| `metrics.py` | `Metric` base class, `METRICS` registry, `Accuracy`, `L0Sparsity`, `ExplainedVariance`. |
| `visualisation.py` | Training-curve plots, `Visualiser` base class, `VISUALISERS` registry, `ConfusionMatrix`. |
| `feature_visualisation.py` | Visualisers for the features of a sparse autoencoder. |
| `tensors.py` | Helpers for model outputs that are a tuple or dict of tensors. |
| `results.py` | Result dataclasses: `EpochResult`, `FoldResult`, `TrainingResult`, `EvaluationResult`. |
| `registry.py` | `Registry`: select classes by string. |

Not to be confused: the root `trainer.py` holds the config-driven
`Trainer(job, config)`, `training.Trainer` is the engine it builds.

## Running MNIST

```
python main.py -c config/mnist_nn.yaml
```

Output goes to `output/<datetime>/job_mnist_nn/`:

- `checkpoints/fold_1/`: weights and loss/metric curves every 
  `checkpoint: every` epochs, `best.pt` and `curves_final.png`.
- `evaluation/`: the visualisers listed under `test: visualisers`.
- `best_model.pt`: weights of the best epoch (of the best fold).
- `predictions.pt`: only if `export: predictions: True`.

## Trainer config

| Key | Meaning |
|-----|---------|
| `mode` | `holdout` (train/val split) or `kfold` (cross validation, needs `kfold: k`). |
| `train`, `val`, `test` | `data`: label of a dataset module. `loader`: DataLoader kwargs overriding `loader_defaults`. |
| `val: fraction` | If `val` uses the same `data` as `train`, this fraction is held out (seeded with `val: seed`). |
| `test: visualisers` | Registered visualiser names, or `[name, {kwargs}]`. |
| `model` | Label of the model module. |
| `epochs` | Epochs per fold. |
| `loss`, `optimiser`, `scheduler` | `name` + `params` of a class in `torch.nn`, `torch.optim` or `torch.optim.lr_scheduler`. Or `module: <label>` for a custom module of the job. |
| `metrics` | Registered metric names, or `[name, {kwargs}]`. |
| `callbacks` | Optional: registered callback names, or `[name, {kwargs}]`. |
| `monitor` | `value` (`loss`, a metric or a loss component) and `goal` (`min`/`max`) deciding the best epoch. Default: `loss`, `min`. |
| `early_stopping` | Optional: `patience`, `min_delta`, on `monitor`. |
| `checkpoint` | Optional: `every`, `save_best`, `plot_curves`. |
| `mixed_precision` | Optional: bfloat16 forward pass. |
| `export: predictions` | Save test predictions and targets. |

## Extending

**Custom loss**: write a module in `modules/` (an `nn.Module`), add it to
`modules/__init__.py`, put it in the job's `modules` and reference it
with `loss: {module: <label>}`. It may return the loss tensor, or
`(loss, {"xy": ..., "wh": ...})` to track named components as well.

**Metric**: subclass `Metric`, decorate with `@METRICS.register("name")`,
make sure the file is imported, then list `name` under `metrics`. The
same metric can be used twice with `[name, {name: other, ...kwargs}]`.

**Visualiser**: subclass `Visualiser`, decorate with
`@VISUALISERS.register("name")`. It receives an `EvaluationResult`
(`predictions`, `targets`, `sample_inputs`, `metrics`, `dataset`) and an
output directory. `dataset` is set unless the dataloader shuffled, so
`dataset[i]` is the datapoint of `predictions[i]`.

**Callback**: subclass `Callback`, decorate with
`@CALLBACKS.register("name")` and list `name` under `callbacks`. Besides
the model, a callback can read the `loss_fn`, `optimiser` and the 
dataloaders of the current fold from the trainer.

**Models with multiple outputs**: the output of a model may be a tensor,
or a tuple or dict of tensors. The loss and metrics receive it as is, and
`EvaluationResult.predictions` has the same structure.

**Different training loop**: subclass the engine and override one 
method, e.g. a detection model that reshapes its output:

```python
class YoloTrainer(Engine):
    def forward(self, batch):
        y_hat, y = super().forward(batch)
        return y_hat.view(-1, GRID_SIZE, GRID_SIZE, 7), y
```

Other overridable methods: `compute_loss`, `compute_metrics`,
`training_step`, `validation_step`, `train_epoch`, `val_epoch`,
`get_fold_indices` (e.g. per-person folds) and `prepare_fold` (e.g.
`dataset.fit_normalisation(train_idx)`).

**Optuna**: add `PruningCallback(report)` to the engine's callbacks, with
`report` taking the monitored validation value.

---

# Config manual

A run is described by one yaml file (`python main.py -c <file>`). It is
validated against `config/config_validation_template.py` before anything
is built, and the error says where in the file the problem is.

```yaml
general:  {...}          # settings shared by all jobs
jobs:                    # a list, jobs run one after the other
    - job:
        name: ...
        modules: [...]   # the building blocks, see "Modules"
        trainer: {...}   # how to train them
```

Notation below: `str`, `int`, `float`, `bool`, `list`, `dict`, `None`
(yaml: empty or `null`). `a | b` means either type is accepted. Write
floats with an exponent with a dot (`1.0e-3`), otherwise YAML reads a
string.

## general

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `output_folder` | str | required | Where results go. `{datetime}` is replaced by the start time of the run. Keep the trailing `/`. |
| `seed` | int | required | Random seed, set before every job. |
| `device` | str | fastest available | Any torch device: `cpu`, `cuda`, `cuda:1`, `mps`, ... |

## job

| Key | Type | Meaning |
|-----|------|---------|
| `name` | str | Name of the job, its results go to `<output_folder>/job_<name>/`. A duplicate name gets a numeric suffix. |
| `modules` | list[dict] | Modules to create, see below. |
| `trainer` | dict | Trainer config, see below. |

### Writing a module

Every entry of `modules` is a single key, the class name of the module,
with its parameters below it. `label` is required for every module and
must be unique in the job, other modules and the trainer refer to it by
this label.

```yaml
modules:
    - MNISTData:
        label: train_data
        root: data/
        train: True
        normalise: True
```

The parameters of all modules are in "Module parameters". Modules that
are not known to the validation template are accepted as long as they
have a `label`.

## trainer

Every key of the trainer section. Unknown keys are rejected.

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `mode` | `holdout` \| `kfold` | `holdout` | How to train, see "Modes". |
| `loader_defaults` | dict | none | Keyword arguments of the torch `DataLoader`, used for train, val and test. |
| `train` | dict | required | `data` (str, label of a dataset module), `loader` (dict, `DataLoader` kwargs that override `loader_defaults`). |
| `val` | dict | required for `holdout` | `data`, `loader`, and `fraction`, `seed`, see "Modes". |
| `test` | dict | required | `data`, `loader`, and `visualisers` (list, see "Registered items"). |
| `model` | str | required | Label of the module to train. |
| `epochs` | int ≥ 1 | required | Number of epochs (per fold in `kfold`). |
| `loss` | dict | required | A loss, see "Loss, optimiser, scheduler". |
| `optimiser` | dict | required | An optimiser. |
| `scheduler` | dict | no scheduler | A learning rate scheduler, stepped every epoch. |
| `metrics` | list | `[]` | Metrics to track, see "Registered items". |
| `monitor` | dict | loss, min | What decides the best epoch, see below. |
| `early_stopping` | dict | off | Stop when the monitored value stops improving. |
| `checkpoint` | dict | off | Save weights and curves while training. |
| `callbacks` | list | `[]` | Extra callbacks, see "Registered items". |
| `mixed_precision` | bool | `False` | Train with bfloat16 autocast. |
| `export` | dict | none | `predictions` (bool): save test predictions and targets to `predictions.pt`. |
| `kfold` | dict | required for `kfold` | `k` (int ≥ 2). |

`train.loader`, `val.loader`, `test.loader` and `loader_defaults` take
any argument of `torch.utils.data.DataLoader`, e.g. `batch_size` (int),
`shuffle` (bool), `num_workers` (int), `pin_memory` (bool),
`drop_last` (bool). Without `shuffle: True` on the train loader the data
is not shuffled, and the visualisers cannot look up the images of the
test set if its loader shuffles.

### Modes

**`holdout`** trains on `train` and validates on `val`. There are two ways
to set up `val`:

```yaml
# 1. A separate dataset module.
val:
    data: test_data

# 2. Hold out a part of the training data (data must equal train.data).
val:
    data: train_data
    fraction: 0.1     # float, 0 < fraction < 1. Required in this case.
    seed: 42          # int, default 42. Makes the split reproducible.
```

**`kfold`** cross validates on `train.data`, so no `val` is needed:

```yaml
mode: kfold
kfold:
    k: 5
```

In both modes the model is evaluated on `test` afterwards.

### Loss, optimiser, scheduler

All three accept the same two forms:

```yaml
# 1. A class of torch (torch.nn, torch.optim, torch.optim.lr_scheduler).
loss:
    name: CrossEntropyLoss       # str, class name
    params: {label_smoothing: 0.1}   # dict | None, kwargs of the class
optimiser:
    name: Adam
    params: {lr: 1.0e-3, weight_decay: 1.0e-5}
scheduler:
    name: StepLR
    params: {step_size: 5}

# 2. A module of this job (for custom ones).
loss:
    module: sae_loss             # str, label of the module
```

The optimiser is given the parameters of the model that require a
gradient, the scheduler is given the optimiser.

### monitor

```yaml
monitor:
    value: loss      # str: `loss`, the name of a metric (`accuracy`), or
                     # a loss component (`mse`, `l1`). Default: loss.
    goal: min        # `min` | `max`. Default: min.
```

The monitored value of the validation data selects the best epoch (saved
as best weights), and is what early stopping watches.

### early_stopping

| Key | Type | Meaning |
|-----|------|---------|
| `patience` | int ≥ 1 | Epochs without improvement before stopping. |
| `min_delta` | float ≥ 0 | Minimal change that counts as improvement. |

### checkpoint

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `every` | int ≥ 1 | 5 | Save weights every n epochs. |
| `save_best` | bool | `True` | Also save the best weights. |
| `plot_curves` | bool | `True` | Plot loss and metric curves. |
| `resume` | bool | `False` | Not supported yet, `True` only logs a warning. |

### Registered items

Metrics, test visualisers and callbacks are chosen by name, either as a
plain string or as `[name, {kwargs}]`. Give the kwarg `name` to rename
an item, so one class can be used twice.

```yaml
metrics: [accuracy]
callbacks:
    - [dead_feature_resampling, {every_n_epochs: 10, until_epoch: 40}]
test:
    visualisers:
        - confusion_matrix
        - [feature_gallery, {top_n: 8}]
        - [feature_gallery, {top_n: 32, name: gallery_big}]
```

**Metrics**

| Name | Kwargs | Meaning |
|------|--------|---------|
| `accuracy` | none | Fraction of correct class predictions. |
| `l0` | none | Average number of active features per datapoint. |
| `explained_variance` | none | Fraction of the variance of the input that the reconstruction explains. |

**Visualisers** (all kwargs optional; `features` is `list[int]`,
`class_names` is `list[str]`, default: the `classes` of the dataset)

| Name | Kwargs | Meaning |
|------|--------|---------|
| `confusion_matrix` | `class_names`, `normalise` (bool) | Predictions against targets. |
| `feature_statistics` | `top_n` (int), `features`, `class_names` | Activation statistics of the most relevant features. |
| `feature_trigger_histogram` | `max_xticks` (int), `class_names` | How often features trigger per class. |
| `feature_gallery` | `top_n` (16), `features`, `class_names`, `n_images` (50), `zoom` (float, 1.5), `bin_interval` (float) | The images that activate a feature most. |
| `feature_summary` | `top_n` (16), `features`, `class_names`, `n_images` (50), `max_groups` (6) | Overview per feature, grouped by class. |
| `feature_contrast` | `top_n` (16), `features`, `class_names`, `n_images` (16), `seed` (0) | Activating against random images. |

`top_n` selects the most relevant features, `features` selects specific
ones by index (and replaces `top_n`).

**Callbacks**

| Name | Kwargs | Meaning |
|------|--------|---------|
| `dead_feature_resampling` | `every_n_epochs` (int, 10), `until_epoch` (int \| None, None), `n_samples` (int, 20000) | Reinitialise sparse autoencoder features that never activate. |

## Module parameters

Besides these, every module takes the `label` (str, unique in the job).
`device` and `logger` are given by the pipeline, never write them.

| Module | Parameter | Type | Default | Meaning |
|--------|-----------|------|---------|---------|
| `Base` | | | | Only a label, for a module that needs nothing else. |
| `MNISTData` | `root` | str | required | Folder to store/load the data. |
| | `train` | bool | required | `True` for the train set, `False` for the test set. |
| | `normalise` | bool | required | Normalise with mean 0.1307 and std 0.3081. |
| | `download` | bool | `True` | Download if not found in `root`. |
| `CIFAR10Data` | `root` | str | required | Folder to store/load the data. |
| | `train` | bool | required | `True` for the train set, `False` for the test set. |
| | `download` | bool | `True` | Download if not found in `root`. |
| `NeuralNetwork` | `network_shape` | list[dict] | required | Layers, in order. Each is `{LayerName: {kwargs}}` with a class of `torch.nn`. Layers without kwargs (`ReLU`) are written with an empty value. |
| `ResNet` | `architecture` | `resnet18` \| `resnet34` \| `resnet50` \| `resnet101` \| `resnet152` | `resnet18` | Which ResNet. Embedding size is 512 for 18/34, 2048 for the others. |
| | `pretrained` | bool | `True` | Start from ImageNet weights. |
| | `frozen` | bool | `True` | Do not train it (and keep it in eval mode). |
| | `input_size` | int \| None | `None` | Resize images to this width and height first. `None` keeps the size. |
| `SparseAutoencoder` | `input_dim` | int | required | Size of the embeddings, must equal the output of the module before it. |
| | `hidden_dim` | int | required | Number of features. |
| `SparseAutoencoderLoss` | `l1_coeff` | float ≥ 0 | required | Weight of the sparsity (L1) penalty. |
| `Chain` | `stages` | list[str] | required | Labels of the modules to run in order. Each output type must equal the next input type. |

Example of a layer list:

```yaml
network_shape:
    - Linear:
        in_features: 784
        out_features: 10
    - ReLU:
    - Dropout:
        p: 0.2
```

### Adding a module or option

- New module: create it in `modules/`, export it in `modules/__init__.py`
  and write it in the config. It is accepted by the validation right
  away. To have its parameters checked too, add a `_module({...})` entry
  to `MODULE_SCHEMAS` in `config/config_validation_template.py`.
- New metric, visualiser or callback: register it, the name is then
  valid in the config without further changes.
- New trainer key: add it to `TRAINER` in the validation template, as
  unknown trainer keys are rejected.

---

# Modules

Modules are the building blocks of a job, created from the `modules` 
list of the config and found by their `label`. Every module is a `Base`.
After all modules of a job exist, each of them gets the chance to refer to
the others by calling `Base.link(modules)`.

| Module | Purpose |
|--------|---------|
| `MNISTData`, `CIFAR10Data` | Datasets. Images are tensors in [0, 1], `CIFAR10Data` also has the names of the `classes`. |
| `NeuralNetwork` | Network of any shape, built from `torch.nn` layers. |
| `ResNet` | Embeds images with any of resnet18/34/50/101/152, optionally pretrained and frozen. |
| `SparseAutoencoder` | Recreates embeddings through a sparse hidden layer, whose activations are the features. |
| `SparseAutoencoderLoss` | Mean squared error plus L1 penalty, tracks `mse` and `l1` as loss components. |
| `Chain` | Runs other modules (by label) one after the other. |

**Chain**: `stages: [backbone, sae]` runs `backbone`, then `sae` on its 
output. The `output_type` of every stage must equal the `input_type` of 
the next, otherwise linking fails with an error. The chain is a regular 
module, so it can be the `model` of the trainer. Frozen parameters are 
left out of the optimiser, and a frozen `ResNet` stays in evaluation mode
while training, so its batch normalisation does not change.

## Sparse autoencoder on CIFAR-10

```
python main.py -c config/cifar10_sae.yaml
```

A frozen, pretrained ResNet embeds CIFAR-10 images, and the sparse 
autoencoder is trained to recreate those embeddings. Only the autoencoder
is trained. The ResNet runs on every image in every epoch, so use a GPU.

Because the autoencoder returns a dict (its `input`, `reconstruction` and
`activations`), the loss can be computed from the output alone and the
metrics `l0` and `explained_variance` track the sparsity and the quality
of the reconstruction.

The `dead_feature_resampling` callback reinitialises features that never 
activate on the training data. Besides the usual output (see Running 
MNIST), the test set is evaluated and these visualisers are saved to 
`evaluation/`, all in terms of CIFAR-10 images and classes:

| Visualiser | Output |
|------------|--------|
| `feature_statistics` | Range, non-zero count and relevance of every feature. |
| `feature_trigger_histogram` | How many features are triggered per image. |
| `feature_gallery` | Per relevant feature: the top activating images grouped by class, above the distribution of its activations. |
| `feature_summary` | All relevant features side by side, one image per class. |
| `feature_contrast` | Per relevant feature: top activating images next to random images that do not activate it. |

The relevant features are the `top_n` features with the highest summed
normalised range and non-zero count, never features that do not activate.
Pick features yourself with `features: [3, 17]`.

To use another embedding model, write a module that outputs embeddings
with an `output_type`, and list it in the `stages` instead of `ResNet`. 
For other data, give the dataset a `classes` attribute and return images 
in [0, 1].
