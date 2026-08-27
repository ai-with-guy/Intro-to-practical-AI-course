#### Linux / macOS:
```bash
./launch.sh
```

#### Windows:
```cmd
launch.bat
```

#### Regenerate the teaching datasets
```bash
uv run python generate_datasets.py
```

The generator uses a fixed seed, so every run produces the same CSV files. Change its
sample sizes, noise, regression frequencies and peaks, or class ratios to experiment
with the model comparisons.
