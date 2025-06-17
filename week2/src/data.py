from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import random


def get_dataset(cfg):
    ds = load_dataset(
        cfg.dataset.name,
        cfg.dataset.version,
        cache_dir="~/experiment_data/datasets/ms_marco",
        split="train")

    rows = []
    for entry in ds:
        q = entry["query"]
        passages = entry["passages"]
        for is_selected, text in zip(passages['is_selected'], passages['passage_text']):
            if is_selected == 1:
                rows.append((q, text))

    all_passages = []
    for p in ds["passages"]:
        all_passages.extend(p["passage_text"])

    class TripletDataset(Dataset):
        def __init__(self, rows, all_passages):
            self.rows = rows
            self.all_passages = all_passages
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            q, pos = self.rows[idx]
            neg = pos
            while neg == pos:
                neg = random.choice(self.all_passages)
            return {"query": q, "pos": pos, "neg": neg}
    
    return TripletDataset(rows, all_passages)


def get_dataloader(cfg):
    def collate_fn(batch):
        return {
            "query": [item["query"] for item in batch],
            "pos": [item["pos"] for item in batch],
            "neg": [item["neg"] for item in batch],
        }
    loader = DataLoader(get_dataset(cfg), batch_size=cfg.dataset.batch_size, shuffle=True, collate_fn=collate_fn)
    return loader
