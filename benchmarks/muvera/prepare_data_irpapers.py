#!/usr/bin/env python3
"""
Prepare IRPAPERS data with ColModernVBERT multi-vector embeddings for MUVERA benchmarking.

IRPAPERS is a visual document benchmark of 3,230 pages from 166 IR papers with 180
needle-in-the-haystack queries. ColModernVBERT encodes page images into ~1000 128-dim
multi-vectors (late interaction), and queries are encoded as text.

This script:
1. Downloads IRPAPERS dataset from HuggingFace (pages + queries)
2. Encodes page images with ColModernVBERT (image encoder)
3. Encodes queries with ColModernVBERT (text encoder)
4. Computes brute-force MaxSim ground truth rankings
5. Saves everything to disk for the benchmark runner

Usage:
    python prepare_data_irpapers.py --output_dir ./data_irpapers --device cuda
"""

import argparse
import base64
import json
import os
import time
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def download_irpapers():
    """Download IRPAPERS dataset from HuggingFace."""
    from datasets import load_dataset

    print("Downloading IRPAPERS pages (docs config)...")
    pages_ds = load_dataset("weaviate/IRPAPERS", "docs", split="train")
    print(f"  Loaded {len(pages_ds)} pages")

    print("Downloading IRPAPERS queries config...")
    queries_ds = load_dataset("weaviate/IRPAPERS", "queries", split="train")
    print(f"  Loaded {len(queries_ds)} queries")

    return pages_ds, queries_ds


def decode_base64_image(b64_string):
    """Decode a base64 string to a PIL Image."""
    img_bytes = base64.b64decode(b64_string)
    return Image.open(BytesIO(img_bytes)).convert("RGB")


def load_colmodernvbert(device="cpu"):
    """Load ColModernVBERT model and processor."""
    from colpali_engine.models import ColModernVBert, ColModernVBertProcessor

    model_id = "ModernVBERT/colmodernvbert"
    print(f"Loading ColModernVBERT from {model_id}...")

    processor = ColModernVBertProcessor.from_pretrained(model_id)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = ColModernVBert.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    print(f"  Model loaded on {device} ({dtype})")
    return model, processor



def encode_pages(model, processor, pages_ds, batch_size=4, device="cpu"):
    """Encode page images into ColModernVBERT multi-vector embeddings."""
    print(f"Encoding {len(pages_ds)} page images with ColModernVBERT...")
    all_embeddings = {}
    page_ids = []
    page_texts = []

    # Build page ID list and extract metadata
    # IRPAPERS columns: pdf_id_x, pdf_name, page_number, base64_str, transcription, ...
    for i, row in enumerate(pages_ds):
        pid = f"{row['pdf_id_x']}_{row['page_number']}"
        page_ids.append(str(pid))
        page_texts.append(row.get("transcription", ""))

    for i in tqdm(range(0, len(pages_ds), batch_size), desc="Encoding pages"):
        batch_rows = [pages_ds[j] for j in range(i, min(i + batch_size, len(pages_ds)))]
        batch_ids = page_ids[i : i + batch_size]

        # Decode images from base64_str column
        images = []
        for row in batch_rows:
            b64 = row.get("base64_str")
            if not b64:
                b64 = row.get("base64_bytes")
            if not b64:
                # Debug: print what we have
                print(f"  WARNING: No image data. base64_str type={type(row.get('base64_str'))}, "
                      f"len={len(str(row.get('base64_str', '')))}, "
                      f"base64_bytes type={type(row.get('base64_bytes'))}")
                raise ValueError(f"No image data found in row: {list(row.keys())}")
            # Handle bytes vs string
            if isinstance(b64, bytes):
                b64 = b64.decode("utf-8")
            img = decode_base64_image(b64)
            images.append(img)

        with torch.no_grad():
            inputs = processor.process_images(images)
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            embeddings = model(**inputs)  # (batch, num_tokens, 128)

            if hasattr(embeddings, "cpu"):
                embeddings = embeddings.cpu().float().numpy()
            else:
                embeddings = np.array(embeddings)

        for j, pid in enumerate(batch_ids):
            emb = embeddings[j]
            # Remove zero-padded tokens
            norms = np.linalg.norm(emb, axis=1)
            valid = emb[norms > 1e-6]
            all_embeddings[pid] = valid.tolist()

    print(f"  Encoded {len(all_embeddings)} pages")
    avg_tokens = np.mean([len(v) for v in all_embeddings.values()])
    print(f"  Average tokens per page: {avg_tokens:.0f}")
    return page_ids, page_texts, all_embeddings


def encode_queries(model, processor, queries_ds, device="cpu"):
    """Encode queries as text with ColModernVBERT text encoder."""
    print(f"Encoding {len(queries_ds)} queries with ColModernVBERT...")
    all_embeddings = {}
    query_ids = []
    query_texts = {}

    for i, row in enumerate(queries_ds):
        qid = str(row.get("query_id") or row.get("id") or f"q_{i}")
        query_ids.append(qid)
        query_texts[qid] = row.get("query", row.get("question", row.get("text", "")))

    batch_size = 32
    for i in tqdm(range(0, len(query_ids), batch_size), desc="Encoding queries"):
        batch_ids = query_ids[i : i + batch_size]
        batch_texts = [query_texts[qid] for qid in batch_ids]

        with torch.no_grad():
            inputs = processor.process_texts(batch_texts)
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            embeddings = model(**inputs)

            if hasattr(embeddings, "cpu"):
                embeddings = embeddings.cpu().float().numpy()
            else:
                embeddings = np.array(embeddings)

        for j, qid in enumerate(batch_ids):
            emb = embeddings[j]
            norms = np.linalg.norm(emb, axis=1)
            valid = emb[norms > 1e-6]
            all_embeddings[qid] = valid.tolist()

    print(f"  Encoded {len(all_embeddings)} queries")
    avg_tokens = np.mean([len(v) for v in all_embeddings.values()])
    print(f"  Average tokens per query: {avg_tokens:.0f}")
    return query_ids, query_texts, all_embeddings



def build_qrels(queries_ds, page_ids):
    """Build qrels mapping from queries dataset.

    IRPAPERS uses needle-in-the-haystack: each query has exactly one relevant page.
    The relevance label is binary (1 = relevant).
    """
    page_id_set = set(page_ids)
    qrels = {}
    for i, row in enumerate(queries_ds):
        qid = str(row.get("query_id") or row.get("id") or f"q_{i}")
        # Try to construct the page ID the same way as encode_pages
        pdf_id = row.get("pdf_id_x") or row.get("pdf_id")
        page_num = row.get("page_number")
        if pdf_id is not None and page_num is not None:
            relevant_page = f"{pdf_id}_{page_num}"
            if relevant_page in page_id_set:
                qrels[qid] = {relevant_page: 1}
                continue
        # Fallback: try any column that looks like a page reference
        for key in row.keys():
            if "page" in key.lower() or "doc" in key.lower() or "relevant" in key.lower():
                val = row[key]
                if val is not None and str(val) in page_id_set:
                    qrels[qid] = {str(val): 1}
                    break
    print(f"  Built qrels for {len(qrels)} / {len(queries_ds)} queries")
    return qrels


def compute_ground_truth(query_ids, query_embeddings, page_ids, page_embeddings, device="cpu"):
    """Compute brute-force MaxSim rankings for all queries."""
    print(f"Computing brute-force MaxSim ground truth on {device}...")
    ground_truth = {}
    dev = torch.device(device)

    # Pre-convert page embeddings to tensors
    page_tensors = {}
    for pid in tqdm(page_ids, desc="Loading page tensors"):
        page_tensors[pid] = torch.tensor(
            page_embeddings[pid], dtype=torch.float32, device=dev
        )

    for qid in tqdm(query_ids, desc="MaxSim ground truth"):
        if qid not in query_embeddings:
            continue
        q = torch.tensor(query_embeddings[qid], dtype=torch.float32, device=dev)
        scores = []
        for pid in page_ids:
            d = page_tensors[pid]
            sim = q @ d.T  # (nq, nd)
            score = float(sim.max(dim=1).values.sum().cpu())
            scores.append((pid, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        ground_truth[qid] = scores

    return ground_truth


def save_data(output_dir, page_ids, page_texts, page_embeddings,
              query_ids, query_texts, query_embeddings, qrels, ground_truth):
    """Save all prepared data to disk."""
    os.makedirs(output_dir, exist_ok=True)

    print("Saving document embeddings...")
    doc_data = {}
    for i, pid in enumerate(page_ids):
        doc_data[pid] = {
            "text": page_texts[i] if i < len(page_texts) else "",
            "embeddings": page_embeddings[pid],
        }
    with open(os.path.join(output_dir, "doc_embeddings.json"), "w") as f:
        json.dump(doc_data, f)

    print("Saving query embeddings...")
    query_data = {}
    for qid in query_ids:
        query_data[qid] = {
            "text": query_texts.get(qid, ""),
            "embeddings": query_embeddings[qid],
        }
    with open(os.path.join(output_dir, "query_embeddings.json"), "w") as f:
        json.dump(query_data, f)

    with open(os.path.join(output_dir, "qrels.json"), "w") as f:
        json.dump(qrels, f)

    print("Saving ground truth rankings...")
    gt_serializable = {}
    for qid, rankings in ground_truth.items():
        gt_serializable[qid] = [
            {"doc_id": pid, "score": score} for pid, score in rankings[:1000]
        ]
    with open(os.path.join(output_dir, "ground_truth_maxsim.json"), "w") as f:
        json.dump(gt_serializable, f)

    print(f"All data saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare IRPAPERS data for MUVERA benchmark")
    parser.add_argument("--output_dir", type=str, default="./data_irpapers")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for encoding (cpu/cuda)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for image encoding (keep small, images are large)")
    args = parser.parse_args()

    start = time.time()

    pages_ds, queries_ds = download_irpapers()

    # Inspect dataset columns
    print(f"  Page columns: {pages_ds.column_names}")
    print(f"  Query columns: {queries_ds.column_names}")
    print(f"  Sample page row keys: {list(pages_ds[0].keys())}")
    print(f"  Sample query row keys: {list(queries_ds[0].keys())}")

    model, processor = load_colmodernvbert(device=args.device)

    page_ids, page_texts, page_embeddings = encode_pages(
        model, processor, pages_ds, args.batch_size, args.device
    )
    query_ids, query_texts, query_embeddings = encode_queries(
        model, processor, queries_ds, args.device
    )

    qrels = build_qrels(queries_ds, page_ids)

    ground_truth = compute_ground_truth(
        query_ids, query_embeddings, page_ids, page_embeddings, device=args.device
    )

    save_data(args.output_dir, page_ids, page_texts, page_embeddings,
              query_ids, query_texts, query_embeddings, qrels, ground_truth)

    elapsed = time.time() - start
    print(f"\nData preparation complete in {elapsed:.1f}s")
    print(f"  Pages: {len(page_ids)}")
    print(f"  Queries: {len(query_ids)}")
    print(f"  Token dim: 128 (ColModernVBERT)")
    print(f"  Avg tokens/page: {np.mean([len(v) for v in page_embeddings.values()]):.0f}")


if __name__ == "__main__":
    main()
