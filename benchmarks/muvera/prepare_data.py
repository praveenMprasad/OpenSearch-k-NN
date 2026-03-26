#!/usr/bin/env python3
"""
Prepare nfcorpus data with ColBERTv2 multi-vector embeddings for MUVERA benchmarking.

Steps:
1. Download BeIR nfcorpus dataset
2. Encode documents and queries with ColBERTv2
3. Compute brute-force MaxSim ground truth rankings
4. Save everything to disk for the benchmark runner

Usage:
    python prepare_data.py --output_dir ./data
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm


def download_nfcorpus(data_dir):
    """Download nfcorpus using BeIR."""
    from beir import util as beir_util
    from beir.datasets.data_loader import GenericDataLoader

    dataset = "nfcorpus"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    download_path = os.path.join(data_dir, "datasets")
    data_path = beir_util.download_and_unzip(url, download_path)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    print(f"Loaded nfcorpus: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels")
    return corpus, queries, qrels


def load_colbert_model(device="cpu"):
    """Load ColBERTv2 model."""
    from colbert.infra import ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint

    config = ColBERTConfig(doc_maxlen=300, query_maxlen=32)
    checkpoint = Checkpoint("colbert-ir/colbertv2.0", colbert_config=config)
    return checkpoint


def encode_documents(checkpoint, corpus, batch_size=32, device="cpu"):
    """Encode corpus documents into ColBERTv2 multi-vector embeddings."""
    doc_ids = sorted(corpus.keys())
    doc_texts = []
    for did in doc_ids:
        title = corpus[did].get("title", "")
        text = corpus[did].get("text", "")
        doc_texts.append(f"{title} {text}".strip())

    print(f"Encoding {len(doc_texts)} documents with ColBERTv2...")
    all_embeddings = {}

    for i in tqdm(range(0, len(doc_texts), batch_size), desc="Encoding docs"):
        batch_texts = doc_texts[i : i + batch_size]
        batch_ids = doc_ids[i : i + batch_size]

        with torch.no_grad():
            # keep_dims=True returns padded 3D tensor (batch, max_tokens, dim)
            result = checkpoint.docFromText(batch_texts, bsize=len(batch_texts),
                                            keep_dims=True, to_cpu=True)
            if isinstance(result, tuple):
                token_embs = result[0]
            else:
                token_embs = result

            token_embs = token_embs.numpy()

        for j, did in enumerate(batch_ids):
            # Remove zero-padded tokens (all-zero rows)
            emb = token_embs[j]
            norms = np.linalg.norm(emb, axis=1)
            valid = emb[norms > 1e-6]
            all_embeddings[did] = valid.tolist()

    return doc_ids, all_embeddings


def encode_queries(checkpoint, queries, device="cpu"):
    """Encode queries into ColBERTv2 multi-vector embeddings."""
    query_ids = sorted(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    print(f"Encoding {len(query_texts)} queries with ColBERTv2...")
    all_embeddings = {}

    batch_size = 64
    for i in tqdm(range(0, len(query_texts), batch_size), desc="Encoding queries"):
        batch_texts = query_texts[i : i + batch_size]
        batch_ids = query_ids[i : i + batch_size]

        with torch.no_grad():
            result = checkpoint.queryFromText(batch_texts, bsize=len(batch_texts),
                                              to_cpu=True)
            if isinstance(result, tuple):
                token_embs = result[0]
            else:
                token_embs = result

            token_embs = token_embs.numpy()

        for j, qid in enumerate(batch_ids):
            # Remove zero-padded tokens
            emb = token_embs[j]
            norms = np.linalg.norm(emb, axis=1)
            valid = emb[norms > 1e-6]
            all_embeddings[qid] = valid.tolist()

    return query_ids, all_embeddings


def compute_maxsim(query_emb, doc_emb):
    """Compute MaxSim score between a query and document multi-vector."""
    q = np.array(query_emb)  # (num_q_tokens, dim)
    d = np.array(doc_emb)    # (num_d_tokens, dim)
    # For each query token, find max similarity with any doc token
    sim_matrix = q @ d.T  # (num_q_tokens, num_d_tokens)
    return float(np.sum(np.max(sim_matrix, axis=1)))


def compute_ground_truth(query_ids, query_embeddings, doc_ids, doc_embeddings):
    """Compute brute-force MaxSim rankings for all queries."""
    print("Computing brute-force MaxSim ground truth...")
    ground_truth = {}

    for qid in tqdm(query_ids, desc="MaxSim ground truth"):
        q_emb = query_embeddings[qid]
        scores = []
        for did in doc_ids:
            score = compute_maxsim(q_emb, doc_embeddings[did])
            scores.append((did, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        ground_truth[qid] = scores

    return ground_truth


def save_data(output_dir, corpus, queries, qrels, doc_ids, doc_embeddings,
              query_ids, query_embeddings, ground_truth):
    """Save all prepared data to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save document embeddings
    print("Saving document embeddings...")
    doc_data = {}
    for did in doc_ids:
        doc_data[did] = {
            "text": f"{corpus[did].get('title', '')} {corpus[did].get('text', '')}".strip(),
            "embeddings": doc_embeddings[did],
        }
    with open(os.path.join(output_dir, "doc_embeddings.json"), "w") as f:
        json.dump(doc_data, f)

    # Save query embeddings
    print("Saving query embeddings...")
    query_data = {}
    for qid in query_ids:
        query_data[qid] = {
            "text": queries[qid],
            "embeddings": query_embeddings[qid],
        }
    with open(os.path.join(output_dir, "query_embeddings.json"), "w") as f:
        json.dump(query_data, f)

    # Save qrels
    with open(os.path.join(output_dir, "qrels.json"), "w") as f:
        json.dump(qrels, f)

    # Save ground truth MaxSim rankings
    print("Saving ground truth rankings...")
    gt_serializable = {}
    for qid, rankings in ground_truth.items():
        gt_serializable[qid] = [{"doc_id": did, "score": score} for did, score in rankings[:1000]]
    with open(os.path.join(output_dir, "ground_truth_maxsim.json"), "w") as f:
        json.dump(gt_serializable, f)

    print(f"All data saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare nfcorpus data for MUVERA benchmark")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device for encoding (cpu/cuda/mps)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding")
    args = parser.parse_args()

    start = time.time()

    corpus, queries, qrels = download_nfcorpus(args.output_dir)
    checkpoint = load_colbert_model(device=args.device)

    doc_ids, doc_embeddings = encode_documents(checkpoint, corpus, args.batch_size, args.device)
    query_ids, query_embeddings = encode_queries(checkpoint, queries, args.device)

    ground_truth = compute_ground_truth(query_ids, query_embeddings, doc_ids, doc_embeddings)

    save_data(args.output_dir, corpus, queries, qrels, doc_ids, doc_embeddings,
              query_ids, query_embeddings, ground_truth)

    elapsed = time.time() - start
    print(f"Data preparation complete in {elapsed:.1f}s")
    print(f"  Documents: {len(doc_ids)}")
    print(f"  Queries: {len(query_ids)}")
    print(f"  Token dim: 128 (ColBERTv2)")


if __name__ == "__main__":
    main()
