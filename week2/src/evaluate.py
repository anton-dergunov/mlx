import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb


def get_embeddings(dataloader, encode_fn, device):
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding queries/documents"):
            batch = batch.to(device)
            emb = encode_fn(batch)
            embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)


def compute_mrr(hits):
    if not hits:
        return 0.0
    return 1.0 / hits[0][0]


def compute_map(hits):
    if not hits:
        return 0.0
    ap = 0.0
    num_hits = 0
    for i, (rank, _) in enumerate(hits):
        num_hits += 1
        ap += num_hits / rank
    return ap / len(hits)


def compute_precision_at_k(top_k_labels):
    return np.sum(top_k_labels) / len(top_k_labels)


def compute_recall_at_k(top_k_labels, total_relevant):
    return np.sum(top_k_labels) / total_relevant if total_relevant > 0 else 0.0


def compute_ndcg_at_k(gains, ideal_gains, k=10):
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains[:k]))
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_gains[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(sim_matrix, qrels, k=10):
    mrr = []
    map_scores = []
    precision_at_k = []
    recall_at_k = []
    ndcg_at_k = []

    for qid, rels in tqdm(qrels, desc="Computing metrics"):
        scores = sim_matrix[qid].numpy()
        rel_dict = {docid: score for docid, score in rels}
        ranked = np.argsort(-scores)
        hits = [(rank + 1, rel_dict[docid]) for rank, docid in enumerate(ranked) if docid in rel_dict]

        mrr.append(compute_mrr(hits))
        map_scores.append(compute_map(hits))

        top_k = ranked[:k]
        top_k_labels = [1 if docid in rel_dict else 0 for docid in top_k]
        precision_at_k.append(compute_precision_at_k(top_k_labels))
        recall_at_k.append(compute_recall_at_k(top_k_labels, len(rel_dict)))

        gains = [rel_dict.get(docid, 0) for docid in top_k]
        ideal_gains = sorted(rel_dict.values(), reverse=True)
        ndcg_at_k.append(compute_ndcg_at_k(gains, ideal_gains, k))

    return {
        "MRR": np.mean(mrr),
        "MAP": np.mean(map_scores),
        f"Precision@{k}": np.mean(precision_at_k),
        f"Recall@{k}": np.mean(recall_at_k),
        f"NDCG@{k}": np.mean(ndcg_at_k),
    }


def get_top_results(sim_matrix, qrels, query_texts, doc_texts, k=10):
    top_results = []

    for qid, rels in qrels[:3]:  # TODO Make this configurable
        scores = sim_matrix[qid].numpy()
        ranked = np.argsort(-scores)
        top_docs = ranked[:k]
        doc_scores = [scores[docid] for docid in top_docs]
        doc_rels = [next((rel for doc, rel in rels if doc == docid), 0) for docid in top_docs]
        docs_info = [
            {
                "score": float(score),
                "rel_label": rel_label,
                "text": doc_texts[docid]
            }
            for docid, score, rel_label in zip(top_docs, doc_scores, doc_rels)
        ]
        top_results.append({
            "query_text": query_texts[qid],
            "top_documents": docs_info
        })

    return top_results


def evaluate(query_loader, query_texts, doc_loader, doc_texts, qrels, model, cfg, device, pad_token_idx, batch_size=1024):
    model.to(device)
    model.eval()

    if cfg.model.type == "dual_encoder":
        # Get precomputed embeddings
        query_embeddings = get_embeddings(query_loader, model.encode_query, device)
        doc_embeddings = get_embeddings(doc_loader, model.encode_document, device)

        query_embeddings = query_embeddings.to(device)
        doc_embeddings = doc_embeddings.to(device)

        num_queries = query_embeddings.shape[0]
        sim_matrix = []

        for start in tqdm(range(0, num_queries, batch_size), desc="Computing similarities"):
            end = min(start + batch_size, num_queries)
            q_batch = query_embeddings[start:end]
            sims = torch.matmul(q_batch, doc_embeddings.T)  # shape: (batch_q, num_docs)
            sim_matrix.append(sims.cpu())

        sim_matrix = torch.cat(sim_matrix, dim=0)

    elif cfg.model.type == "cross_encoder":
        all_query_tokens = list(query_loader.dataset)
        all_doc_tokens = list(doc_loader.dataset)

        sim_matrix = []
        for q_idx, query in enumerate(tqdm(all_query_tokens, desc="Cross-encoding query-document pairs")):
            sims = []

            # Convert query to tensor and pad to match documents later
            query_tensor = torch.tensor(query, dtype=torch.long).to(device)

            for doc_start in range(0, len(all_doc_tokens), batch_size):
                doc_batch = all_doc_tokens[doc_start:doc_start + batch_size]

                # Convert all documents in batch to tensors
                doc_batch_tensors = [torch.tensor(doc, dtype=torch.long) for doc in doc_batch]

                # Pad document batch
                padded_docs = pad_sequence(doc_batch_tensors, batch_first=True, padding_value=pad_token_idx).to(device)  # (B, T_doc)

                # Pad query to match document length
                query_len = query_tensor.size(0)
                doc_max_len = padded_docs.size(1)
                if query_len < doc_max_len:
                    pad_len = doc_max_len - query_len
                    padded_query = F.pad(query_tensor, (0, pad_len), value=pad_token_idx)
                else:
                    padded_query = query_tensor[:doc_max_len]  # truncate if needed

                # Repeat query for batch
                query_batch = padded_query.unsqueeze(0).expand(padded_docs.size(0), -1)  # (B, T_doc)

                with torch.no_grad():
                    logits = model(query_batch, padded_docs).squeeze(-1)  # (B,)
                    scores = torch.sigmoid(logits)  # normalize to [0, 1]

                sims.append(scores.cpu())

            sims = torch.cat(sims)  # (num_docs,)
            sim_matrix.append(sims.unsqueeze(0))  # (1, num_docs)

        sim_matrix = torch.cat(sim_matrix, dim=0)  # (num_queries, num_docs)

    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    # Compute top results and metrics
    # TODO (Make the number of queries and top docs configurable)
    top_results = get_top_results(sim_matrix, qrels, query_texts, doc_texts, cfg.eval.k)
    results = compute_metrics(sim_matrix, qrels, cfg.eval.k)

    if cfg.log.wandb:
        rows = []
        for result in top_results:
            query = result["query_text"]
            for idx, doc in enumerate(result["top_documents"], 1):
                rows.append({
                    "query": query,
                    "rank": idx,
                    "score": doc["score"],
                    "rel_label": doc["rel_label"],
                    "doc_text": doc["text"]
                })
        df = pd.DataFrame(rows)
        wandb.log({"top_results": wandb.Table(dataframe=df)})
        wandb.log(results)

    return results, top_results
