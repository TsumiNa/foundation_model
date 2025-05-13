# Example Experiment Logs

This directory is intended to store example logs generated from running the CLI examples in `samples/cli_examples/`.

The structure of these logs will typically follow what PyTorch Lightning's `CSVLogger` (or other loggers) produces. For example:

```
example_logs/
├── basic_run/
│   ├── basic_experiment/
│   │   ├── version_0/
│   │   │   ├── checkpoints/
│   │   │   │   └── epoch=X-step=Y.ckpt
│   │   │   ├── hparams.yaml
│   │   │   └── metrics.csv
│   │   └── version_1/
│   │       └── ...
│   └── ...
├── override_run/
│   └── ...
└── scaling_law_regression_1/
    ├── ratio_0.1/
    │   └── ...
    ├── ratio_0.2/
    │   └── ...
    └── ...
```

These logs can be used as input for `samples/helper_tools/scaling_law_analyzer.py` to demonstrate its functionality.
