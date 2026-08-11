# Installation

```bash
conda create -n f2net python=3.11 -y
conda activate f2net
pip install -r requirements.txt
```

Requirements:

- Python >= 3.9
- PyTorch >= 2.0 (CUDA build recommended for UHR inputs)
- `timm`, `einops`, `numpy`, `pillow`

Optional:

- `triton` — accelerates the selective-scan operators in `models/vmamba/`;
  without it (or on CPU) a pure PyTorch fallback is used automatically.
- `pytest` — to run the unit tests: `pytest tests/`
