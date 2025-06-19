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


# FIXME rename to say that it is for train
def load_or_tokenize_dataset(cfg):
    processed_data_path = os.path.join(cfg.dataset.cache_dir, "train.plk")

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

    for entry in tqdm(ds, desc="Tokenizing train dataset"):
        query_tokens = tokenize(entry["query"])
        all_queries.append(query_tokens)

        for passage in entry["passages"]["passage_text"]:
            all_passages.append((tokenize(passage), len(all_queries) - 1))

    print(f"Saving tokenized data to: {processed_data_path}")
    with open(processed_data_path, "wb") as f:
        pickle.dump((all_queries, all_passages), f)

    return all_queries, all_passages


def extract_eval_data(cfg, split):
    processed_data_path = os.path.join(cfg.dataset.cache_dir, f"{split}.pkl")

    if os.path.exists(processed_data_path):
        print(f"Loading cached tokenized data from: {processed_data_path}")
        with open(processed_data_path, "rb") as f:
            queries_orig, queries_tokenized, docs_orig, docs_tokenized, qrels = pickle.load(f)
        return queries_orig, queries_tokenized, docs_orig, docs_tokenized, qrels

    ds = load_dataset(
        cfg.dataset.name,
        cfg.dataset.version,
        cache_dir=cfg.dataset.cache_dir,
        split="validation")

    queries_orig = []
    queries_tokenized = []
    docs_orig = []
    docs_tokenized = []
    doc_set = {}
    qrels = []

    for query_id, entry in enumerate(tqdm(ds, desc=f"Tokenizing {split} dataset")):
        query = entry["query"]
        queries_orig.append(query)
        queries_tokenized.append(tokenize(query))

        passage_texts = entry["passages"]["passage_text"]
        is_selected = entry["passages"]["is_selected"]

        relevant_docs = []
        for i, (p_text, selected) in enumerate(zip(passage_texts, is_selected)):
            tokenized = tokenize(p_text)
            doc_key = tuple(tokenized)
            if doc_key not in doc_set:
                doc_id = len(docs_tokenized)
                doc_set[doc_key] = doc_id
                docs_orig.append(p_text)
                docs_tokenized.append(tokenized)
            else:
                doc_id = doc_set[doc_key]

            relevant_docs.append((doc_id, 2 if selected else 1))

        if relevant_docs:
            qrels.append((query_id, relevant_docs))

    print(f"Saving tokenized data to: {processed_data_path}")
    with open(processed_data_path, "wb") as f:
        pickle.dump((queries_orig, queries_tokenized, docs_orig, docs_tokenized, qrels), f)

    return queries_orig, queries_tokenized, docs_orig, docs_tokenized, qrels


class BaseTextDataset(Dataset):
    def __init__(self, word2idx):
        self.word2idx = word2idx
        self.unk_idx = word2idx[UNK_TOKEN]

    def _encode(self, words):
        return [self.word2idx.get(w, self.unk_idx) for w in words]


class TextDataset(BaseTextDataset):
    def __init__(self, texts, word2idx):
        super().__init__(word2idx)
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self._encode(self.texts[idx])


class TripletDataset(BaseTextDataset):
    def __init__(self, queries, passages, word2idx):
        super().__init__(word2idx)
        self.queries = queries
        self.passages = passages

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


def collate_fn(pad_idx, batch):
    return pad_sequence([torch.tensor(x) for x in batch], batch_first=True, padding_value=pad_idx)


def triplet_collate_fn(padding_value, batch):
    def pad(field_name):
        return collate_fn(padding_value, [sample[field_name] for sample in batch])

    return {
        "query": pad("query"),
        "pos": pad("pos"),
        "neg": pad("neg")
    }


# FIXME rename to say that it is for train
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


def get_evaluation_data(cfg, split, word2idx):
    queries_orig, queries_tokenized, docs_orig, docs_tokenized, qrels = extract_eval_data(cfg, split)

    query_dataset = TextDataset(queries_tokenized, word2idx)
    doc_dataset = TextDataset(docs_tokenized, word2idx)

    collate = partial(collate_fn, word2idx[PAD_TOKEN])
    query_loader = DataLoader(
        query_dataset,
        batch_size=cfg.dataset.batch_size,
        collate_fn=collate)
    doc_loader = DataLoader(
        doc_dataset,
        batch_size=cfg.dataset.batch_size,
        collate_fn=collate)
    
    return query_loader, queries_orig, doc_loader, docs_orig, qrels
