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
| `callbacks.py` | `Callback` base class, `EarlyStoppingCallback`, `PruningCallback`, `CheckpointCallback`. |
| `early_stopper.py` | `EarlyStopper`, used by `EarlyStoppingCallback`. |
| `metrics.py` | `Metric` base class, `METRICS` registry, `Accuracy`. |
| `visualisation.py` | Training-curve plots, `Visualiser` base class, `VISUALISERS` registry, `ConfusionMatrix`. |
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
(`predictions`, `targets`, `sample_inputs`, `metrics`) and an output
directory.

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
