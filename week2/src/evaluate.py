import torch
from tqdm import tqdm
import numpy as np
import cProfile
import pstats


def get_embeddings(dataloader, model, device, is_query):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding " + ("queries" if is_query else "documents")):
            batch = batch.to(device)
            emb = model.encode_query(batch) if is_query else model.encode_doc(batch)
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


def get_top_results(sim_matrix, query_texts, doc_texts, k=10):
    selected_queries = range(3)  # TODO Make this configurable

    top_results = []

    for qid in selected_queries:
        scores = sim_matrix[qid].numpy()
        ranked = np.argsort(-scores)
        top_docs = ranked[:k]
        doc_scores = [scores[docid] for docid in top_docs]
        docs_info = [
            {
                "score": float(score),
                "text": doc_texts[docid]
            }
            for docid, score in zip(top_docs, doc_scores)
        ]
        top_results.append({
            "query_text": query_texts[qid],
            "top_documents": docs_info
        })

    return top_results


def evaluate(query_loader, query_texts, doc_loader, doc_texts, qrels, model, device, batch_size=1024):
    query_embeddings = get_embeddings(query_loader, model, device, is_query=True)
    doc_embeddings = get_embeddings(doc_loader, model, device, is_query=False)

    query_embeddings = query_embeddings.to(device)
    doc_embeddings = doc_embeddings.to(device)

    num_queries = query_embeddings.shape[0]
    sim_matrix = []

    for start in tqdm(range(0, num_queries, batch_size), desc="Computing similarities"):
        end = min(start + batch_size, num_queries)
        q_batch = query_embeddings[start:end]
        sims = torch.matmul(q_batch, doc_embeddings.T)
        sim_matrix.append(sims.cpu())

    sim_matrix = torch.cat(sim_matrix, dim=0)

    top_results = get_top_results(sim_matrix, query_texts, doc_texts, k=10)
    results = compute_metrics(sim_matrix, qrels, k=10)

    return results, top_results
