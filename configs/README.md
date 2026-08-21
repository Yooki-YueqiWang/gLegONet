# Configurations

Each JSON file uses schema version 1 and contains an `entrypoint` plus an `arguments` object. Run it with:

```bash
python scripts/run_config.py --config configs/paper/<name>.json
```

Values stated in the manuscript are kept here rather than as hidden Python defaults. Training files additionally include provenance metadata that separates manuscript-stated settings from implementation controls not reported in the paper.
