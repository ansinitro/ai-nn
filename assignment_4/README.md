# Assignment 4: 1D CNN Classification

## Run

```bash
uv run python assignment_4/assignment_4_solution.py
```

## Outputs

- `assignment_4/cnn_classifier.h5` - saved best CNN classifier
- `assignment_4/figures/` - generated charts and diagrams
- `assignment_4/results_summary.json` - final metrics and metadata
- `assignment_4/comparison_metrics.csv` - classification comparison with baseline and previous assignments
- `assignment_4/cnn_search_metrics.csv` - CNN hyperparameter search results
- `assignment_4/presentation.html` - slide-based presentation site

## Note About The Target

The assignment text refers to `Appliances`, but the provided processed dataset from `assignment_1/data/processed/processed_data.csv` contains the weather target used in assignments 2 and 3.

To keep the pipeline comparable with earlier work, the binary label is built from `T (degC)` using the global median threshold.
