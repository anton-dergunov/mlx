import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from functools import partial
from tqdm import tqdm
import pickle
import random
import re
import os

from embeddings import UNK_TOKEN, PAD_TOKEN


def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def load_or_tokenize_dataset(cfg):
    processed_data_path = os.path.join(cfg.dataset.cache_dir, "cache.plk")

    if os.path.exists(processed_data_path):
        print(f"Loading cached tokenized data from: {processed_data_path}")
        with open(processed_data_path, "rb") as f:
            all_queries, all_passages = pickle.load(f)
        return all_queries, all_passages

    print("Tokenizing and saving dataset...")
    ds = load_dataset(
        cfg.dataset.name,
        cfg.dataset.version,
        cache_dir=cfg.dataset.cache_dir,
        split="train")

    all_queries = []
    all_passages = []

    for entry in tqdm(ds):
        query_tokens = tokenize(entry["query"])
        all_queries.append(query_tokens)

        for passage in entry["passages"]["passage_text"]:
            all_passages.append((tokenize(passage), len(all_queries) - 1))

    print(f"Saving tokenized data to: {processed_data_path}")
    with open(processed_data_path, "wb") as f:
        pickle.dump((all_queries, all_passages), f)

    return all_queries, all_passages


class TripletDataset(Dataset):
    def __init__(self, triplets, word2idx):
        self.triplets = triplets
        self.word2idx = word2idx
        self.pad_idx = word2idx["<PAD>"]
        self.unk_idx = word2idx["<UNK>"]

    def encode(self, words):
        return [self.word2idx.get(w, self.unk_idx) for w in words]

    def __getitem__(self, idx):
        q, p, n = self.triplets[idx]
        return {
            "query": self.encode(q),
            "pos": self.encode(p),
            "neg": self.encode(n)
        }

    def __len__(self):
        return len(self.triplets)


class TripletDataset(Dataset):
    def __init__(self, queries, passages, word2idx):
        self.queries = queries
        self.passages = passages
        self.word2idx = word2idx
        self.unk_idx = word2idx[UNK_TOKEN]

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, idx):
        pos_passage, query_id = self.passages[idx]

        # Sample a negative passage not from the same query
        while True:
            neg_idx = random.randint(0, len(self.passages) - 1)
            neg_passage, neg_query_id = self.passages[neg_idx]
            if neg_query_id != query_id:
                break

        return {
            "query": self._encode(self.queries[query_id]),
            "pos": self._encode(pos_passage),
            "neg": self._encode(neg_passage)
        }

    def _encode(self, words):
        return [self.word2idx.get(w, self.unk_idx) for w in words]


def triplet_collate_fn(padding_value, batch):
    def pad(field_name):
        batch_field = [torch.tensor(sample[field_name]) for sample in batch]
        return pad_sequence(batch_field, batch_first=True, padding_value=padding_value)

    return {
        "query": pad("query"),
        "pos": pad("pos"),
        "neg": pad("neg")
    }


def get_dataloader(cfg, word2idx):
    queries, passages = load_or_tokenize_dataset(cfg)

    dataset = TripletDataset(queries, passages, word2idx)

    # TODO Introduce custom bucketed sampler for triplets for effective padding
    collate = partial(triplet_collate_fn, word2idx[PAD_TOKEN])
    return DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        collate_fn=collate)
