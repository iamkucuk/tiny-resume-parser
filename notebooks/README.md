# Notebooks

| Notebook | What it does |
|---|---|
| `colab_train_lora.ipynb` | The first LoRA fine-tune of InternVL3-1B on Colab (2026-10-11), with a live training dashboard (`colab_utils.py`) |
| `colab_speed_test.ipynb` | CPU / GPU speed of the three students on validation pages (the CPU column of the results table) |

Both `%cd` into `/content/cv_parser` (the repository cloned there) and read data from Google Drive. Run `pip install -e .` in that folder first: the cells call `python -m tiny_resume_parser...`.
