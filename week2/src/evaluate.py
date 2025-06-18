import torch
from tqdm import tqdm
import numpy as np


def get_embeddings(dataloader, model, device, is_query):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding " + ("queries" if is_query else "documents")):
            batch = batch.to(device)
            emb = model.encode_query(batch) if is_query else model.encode_doc(batch)
            embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)


def dcg_at_k(relevance_scores, k=10):
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores[:k]))


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

    # Select first 3 queries
    top_k = 10
    selected_queries = range(3)

    top_results = []
    for qid in selected_queries:
        scores = sim_matrix[qid].numpy()
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i], reverse=True)
        top_docs = ranked[:top_k]
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

    # FIXME break this into separate functions

    mrr = []
    map_scores = []
    precision_at_10 = []
    recall_at_10 = []
    ndcg_at_10 = []

    for qid, rels in tqdm(qrels, desc="Computing metrics"):
        scores = sim_matrix[qid].numpy()
        rel_dict = {docid: score for docid, score in rels}
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i], reverse=True)

        hits = []
        for rank, docid in enumerate(ranked):
            if docid in rel_dict:
                hits.append((rank + 1, rel_dict[docid]))

        # MRR
        if hits:
            rank_positions = [r for r, _ in hits]
            mrr.append(1.0 / rank_positions[0])

        # MAP
        ap = 0.0
        num_hits = 0
        for i, (rank, _) in enumerate(hits):
            num_hits += 1
            ap += num_hits / rank
        if hits:
            map_scores.append(ap / len(hits))

        # Precision@10 and Recall@10
        top10 = ranked[:10]
        rels_at_10 = [1 if docid in rel_dict else 0 for docid in top10]
        precision_at_10.append(np.sum(rels_at_10) / 10)
        if rel_dict:
            recall_at_10.append(np.sum(rels_at_10) / len(rel_dict))

        # NDCG@10
        gains = [rel_dict.get(docid, 0) for docid in top10]
        ideal_gains = sorted([v for v in rel_dict.values()], reverse=True)
        dcg = dcg_at_k(gains, k=10)
        idcg = dcg_at_k(ideal_gains, k=10)
        ndcg_at_10.append(dcg / idcg if idcg > 0 else 0.0)

    results = {
        "MRR": np.mean(mrr),
        "MAP": np.mean(map_scores),
        "Precision@10": np.mean(precision_at_10),
        "Recall@10": np.mean(recall_at_10),
        "NDCG@10": np.mean(ndcg_at_10),
    }

    return results, top_results
