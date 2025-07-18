# Minimal experiment template

The template uses:
- `uv` for package management
- `pipreqs` for `requirements.txt`
- `wandb` for experiment tracking (Weights & Biases)
- `omegaconf` for YAML config management

### Install UV
```bash
brew install uv
```
or
```bash
curl -LsSf https://astral.sh/uv/install.sh | less
```

### Create virtual environment
```bash
uv venv
```

### Activate the environment
```bash
source .venv/bin/activate
```

### Create `requirements.txt`:
```bash
uv pip install pipreqs
pipreqs . --force --savepath=requirements.txt
```
- `--extra-ext=.ipynb`: to also scan `.ipynb` notebooks
- `--mode no-pin`: to drop version pinning

Alternative (lists all packages):
```bash
uv pip freeze > requirements.txt
```

### Install the requirements
```bash
uv pip install -r requirements.txt
```

Note: It is important to run as `uv pip` to ensure the correct environment is used.

### Run (after env is activated)
```bash
python src/main.py --config config/base.yaml
```

### TODO
- Also create scripts for remote execution.
  rsync -avz ./my_project user@remote:/home/user/my_project
  ssh user@remote
  cd my_project
  uv venv
  uv pip install -r requirements.txt
  TBC
- Also instructions for GPU monitoring.
- Instructions for keeping the experiment run results.
- Where to keep the experiment data?
- Instructions for notebooks.
- Add useful utils (such as get_device).
- Decribe basic (default) and custom config.

### References
- [Chat with ChatGPT](https://chatgpt.com/share/68515cef-c980-8002-9b31-e3bbbcde1ea5)
