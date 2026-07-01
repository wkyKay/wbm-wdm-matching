# shared

Common utilities used by multiple retrieval methods.

## wm38k

`shared/wm38k` owns the dataset split and manifest logic for MixedWM38K. Methods such as `partial_match` and `Wafer-DenseIR` should read the generated manifests instead of creating their own random splits.

Files:

- `io.py`: loads `Wafer_Map_Datasets.npz`, filters valid labeled samples, and builds label signatures.
- `split.py`: creates stratified `train / valid / test` splits by exact multi-label signature.
- `query.py`: samples a fixed stratified query subset from the test split.
- `manifest.py`: writes and reads split/query CSV manifests.
- `cli_build_split.py`: command-line entry point for generating frozen split and query manifests.
- `__init__.py`: exports the shared WM38K helper functions.

Typical output:

- `artifacts/splits/wm38k_seed2026_sig_70_10_20.csv`
- `artifacts/splits/wm38k_seed2026_test_queries_2000.csv`
- `artifacts/splits/wm38k_seed2026_meta.json`
