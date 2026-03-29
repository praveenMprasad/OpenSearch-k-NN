# MUVERA Benchmark Results

## Overview

This benchmark evaluates the MUVERA processors for OpenSearch using multi-vector
embeddings. MUVERA converts variable-length multi-vector representations into
fixed-size single vectors (FDE) that approximate MaxSim scoring, enabling fast
ANN retrieval while preserving multi-vector quality.

## Datasets

| Dataset | Model | Docs | Queries | Avg vectors/doc | Dim |
|---------|-------|------|---------|-----------------|-----|
| nfcorpus | ColBERTv2 | 3,633 | 323 | ~30 | 128 |
| SciFact | ColBERTv2 | 5,183 | 300 | ~30 | 128 |
| IRPAPERS | ColModernVBERT | 3,230 | 180 | ~1,011 | 128 |

## MUVERA Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| dim | 128 | Token vector dimension |
| k_sim | 5 | SimHash hyperplanes (2^5 = 32 clusters) |
| dim_proj | 16 | Projected dimension per cluster |
| r_reps | 20 | Independent repetitions |
| FDE dimension | 10,240 | Output: 20 x 32 x 16 |

## Index Configuration

- Engine: faiss HNSW, space type: innerproduct
- m: 16, ef_construction: 512, ef_search: 512
- 1 shard, 0 replicas, force merged to 1 segment

## Results: nfcorpus (3,633 docs, 323 queries)

| Approach | NDCG@1 | NDCG@5 | NDCG@10 | % of Exact |
|----------|--------|--------|---------|------------|
| Exact MaxSim (brute-force) | 0.483 | 0.386 | 0.344 | 100% |
| MUVERA + MaxSim rerank (4x) | 0.449 | 0.349 | 0.311 | 90.3% |
| Mean pool + MaxSim rerank | 0.251 | 0.184 | 0.144 | 42.0% |

## Results: SciFact (5,183 docs, 300 queries)

| Approach | NDCG@1 | NDCG@5 | NDCG@10 | % of Exact |
|----------|--------|--------|---------|------------|
| Exact MaxSim (brute-force) | 0.597 | 0.674 | 0.692 | 100% |
| MUVERA + MaxSim rerank (4x) | 0.590 | 0.655 | 0.671 | 97.1% |
| Mean pool + MaxSim rerank | 0.350 | 0.359 | 0.363 | 52.4% |

## Results: IRPAPERS (3,230 pages, 180 queries, ColModernVBERT)

IRPAPERS is a visual document benchmark of 166 IR papers (3,230 pages) with 180
needle-in-the-haystack queries. ColModernVBERT encodes page images into ~1,011
multi-vectors per page. Metric is Recall@K (single relevance per query).

| Approach | R@1 | R@5 | R@10 | R@20 | % of Exact (R@1) |
|----------|-----|-----|------|------|-------------------|
| Exact MaxSim (brute-force) | 40.6% | 76.7% | 83.9% | 89.4% | 100% |
| MUVERA + MaxSim rerank (4x) | 37.2% | 66.7% | 73.3% | 78.3% | 91.7% |

## Key Findings

- **MUVERA + rerank (4x) recovers 90-97% of exact MaxSim quality** on text datasets
  (nfcorpus, SciFact) with ColBERTv2 embeddings (~30 vectors per doc).
- **On IRPAPERS (visual documents, ~1,011 vectors per page), MUVERA + rerank (4x)
  recovers 91.7% of exact MaxSim at R@1**, demonstrating that MUVERA works with
  high-vector-count documents from vision models like ColModernVBERT.
- **Mean pooling retains only 42-52%** of quality, demonstrating why MUVERA is needed:
  naive single-vector approaches discard fine-grained token-level information.

## Approach Details

| Approach | How it works |
|----------|-------------|
| Exact MaxSim | Brute-force MaxSim over all docs (offline, no ANN) |
| MUVERA + rerank (4x) | FDE ANN → top 4x candidates → MaxSim rerank → top K |
| Mean pool + rerank | Mean-pool query → KNN on mean vector (k=100) → MaxSim rerank → top K |

## Reproducibility

```bash
# Build and start OpenSearch with the k-NN plugin
cd k-NN && ./gradlew assemble -x test -x integTest

# Set up Python environment
python3 -m venv benchmarks/muvera/.venv
benchmarks/muvera/.venv/bin/pip install -r benchmarks/muvera/requirements.txt

# Prepare data
python3 benchmarks/muvera/prepare_data.py \
    --output_dir benchmarks/muvera/data --dataset nfcorpus --device cuda

# Run benchmark
python3 -u benchmarks/muvera/run_benchmark.py \
    --data_dir benchmarks/muvera/data --dataset nfcorpus --cleanup_first \
    --output benchmarks/muvera/benchmark_results.json
```
