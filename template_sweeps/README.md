# Example ML project with W&B sweep + YAML config

Run standalone:

```bash
python src/main.py --config config/base.yaml
```

Run sweep:

```bash
wandb sweep config/sweep.yaml
wandb agent your_entity/ms-marco-ranking/your_sweep_id
```
