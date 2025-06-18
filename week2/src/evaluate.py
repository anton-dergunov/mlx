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


def evaluate(query_loader, doc_loader, qrels, model, device):
    query_embeddings = get_embeddings(query_loader, model, device, is_query=True)
    doc_embeddings = get_embeddings(doc_loader, model, device, is_query=False)

    sim_matrix = torch.matmul(query_embeddings, doc_embeddings.T)

    ranks = []
    mrr = []
    map_scores = []

    for qid, rels in qrels:
        scores = sim_matrix[qid].numpy()
        rel_dict = {docid: score for docid, score in rels}
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        hits = []
        for rank, docid in enumerate(ranked):
            if docid in rel_dict:
                hits.append((rank + 1, rel_dict[docid]))

        if hits:
            rank_positions = [r for r, _ in hits]
            reciprocal = 1.0 / rank_positions[0]
            mrr.append(reciprocal)

            ap = 0.0
            num_hits = 0
            for i, (rank, _) in enumerate(hits):
                num_hits += 1
                ap += num_hits / rank
            map_scores.append(ap / len(hits))

            ranks.append(rank_positions[0])

    results = {
        "MRR": np.mean(mrr),
        "MAP": np.mean(map_scores),
        "Recall@10": np.mean([any(r <= 10 for r, _ in sorted([(rank, rel_dict.get(rank, 0)) for rank in range(len(scores)) if rank in rel_dict], key=lambda x: x[0])) for qid, rels in qrels]),
    }

    return results
