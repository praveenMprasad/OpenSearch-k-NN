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

## MUVERA Parameters (nfcorpus / SciFact)

| Parameter | Value | Description |
|-----------|-------|-------------|
| dim | 128 | ColBERTv2 token vector dimension |
| k_sim | 5 | SimHash hyperplanes (2^5 = 32 clusters) |
| dim_proj | 16 | Projected dimension per cluster |
| r_reps | 20 | Independent repetitions |
| FDE dimension | 10,240 | Output: 20 x 32 x 16 |

## Index Configuration

- Engine: faiss HNSW, space type: innerproduct
- m: 16, ef_construction: 512, ef_search: 512
- 1 shard, 0 replicas, force merged to 1 segment

## Results: nfcorpus (3,633 docs, 323 queries)

| Approach | NDCG@1 | NDCG@5 | NDCG@10 | % of Exact | Avg Latency |
|----------|--------|--------|---------|------------|-------------|
| Exact MaxSim (brute-force) | 0.483 | 0.386 | 0.344 | 100% | offline |
| MUVERA + MaxSim rerank (4x) | 0.449 | 0.349 | 0.311 | 90.3% | 1,317ms |
| MUVERA FDE-only (no rerank) | 0.345 | 0.276 | 0.249 | 72.4% | 98ms |
| Mean pool + MaxSim rerank | 0.251 | 0.184 | 0.144 | 42.0% | 485ms |

## Results: SciFact (5,183 docs, 300 queries)

| Approach | NDCG@1 | NDCG@5 | NDCG@10 | % of Exact | Avg Latency |
|----------|--------|--------|---------|------------|-------------|
| Exact MaxSim (brute-force) | 0.597 | 0.674 | 0.692 | 100% | offline |
| MUVERA + MaxSim rerank (4x) | 0.590 | 0.655 | 0.671 | 97.1% | 1,392ms |
| MUVERA FDE-only (no rerank) | 0.413 | 0.499 | 0.528 | 76.3% | 107ms |
| Mean pool + MaxSim rerank | 0.350 | 0.359 | 0.363 | 52.4% | 545ms |

## Key Findings

- **MUVERA + rerank (4x) recovers 90-97% of exact MaxSim quality** across both datasets,
  with 4x oversampling (fetches 40 FDE candidates, reranks with MaxSim).
- **MUVERA FDE-only achieves 72-76%** of exact MaxSim at ~100ms per query — pure ANN
  search on the FDE vector with no MaxSim reranking. This is the true MUVERA approximation
  quality without any late interaction scoring.
- **Mean pooling retains only 42-52%** of quality, demonstrating why MUVERA is needed:
  naive single-vector approaches discard fine-grained token-level information.
- FDE-only is 13x faster than MUVERA+rerank, showing the latency cost of MaxSim reranking.

## Approach Details

| Approach | How it works |
|----------|-------------|
| Exact MaxSim | Brute-force MaxSim over all docs (offline, no ANN) |
| MUVERA + rerank (4x) | FDE ANN → top 40 candidates → MaxSim rerank → top 10 |
| MUVERA FDE-only | Client-side FDE encoding → KNN on FDE field → top 10 (no MaxSim) |
| Mean pool + rerank | Mean-pool query → KNN on mean vector (k=100) → MaxSim rerank → top 10 |

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
