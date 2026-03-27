# MUVERA Benchmark Results

## Overview

This benchmark evaluates the MUVERA processors for OpenSearch on two BeIR datasets
using ColBERTv2 multi-vector embeddings. MUVERA converts variable-length multi-vector
representations into fixed-size single vectors (FDE) that approximate MaxSim scoring,
enabling fast ANN retrieval while preserving multi-vector quality.

## MUVERA Parameters

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

| Approach | NDCG@1 | NDCG@5 | NDCG@10 | % of Exact MaxSim |
|----------|--------|--------|---------|-------------------|
| Exact MaxSim (brute-force) | 0.483 | 0.386 | 0.344 | 100% |
| MUVERA + MaxSim rerank (4x) | 0.474 | 0.379 | 0.339 | 98.5% |
| MUVERA-only | 0.464 | 0.369 | 0.329 | 95.6% |
| Mean pool + MaxSim rerank | 0.249 | 0.183 | 0.145 | 42.2% |

## Results: SciFact (5,183 docs, 300 queries)

| Approach | NDCG@1 | NDCG@5 | NDCG@10 | % of Exact MaxSim |
|----------|--------|--------|---------|-------------------|
| Exact MaxSim (brute-force) | 0.597 | 0.674 | 0.692 | 100% |
| MUVERA + MaxSim rerank (4x) | 0.600 | 0.670 | 0.683 | 98.7% |
| MUVERA-only | 0.597 | 0.665 | 0.679 | 98.1% |
| Mean pool + MaxSim rerank | 0.360 | 0.369 | 0.373 | 53.9% |

## Key Findings

- **MUVERA + rerank recovers 98-99% of exact MaxSim quality** across both datasets.
- **MUVERA-only achieves 96-98%** without any reranking, showing the FDE encoding
  alone captures most of the multi-vector signal.
- **Mean pooling retains only 42-54%** of quality, demonstrating why MUVERA is needed:
  naive single-vector approaches discard fine-grained token-level information.
- On SciFact, MUVERA-only NDCG@1 ties exact MaxSim (0.597), and MUVERA+rerank
  slightly exceeds it (0.600) due to the oversampling effect.

## Reproducibility

```bash
# Build and start OpenSearch with the k-NN plugin
cd k-NN && ./gradlew assemble -x test -x integTest

# Set up Python environment
python3 -m venv benchmarks/muvera/.venv
benchmarks/muvera/.venv/bin/pip install -r benchmarks/muvera/requirements.txt

# Prepare data
PATH="benchmarks/muvera/.venv/bin:$PATH" python3 benchmarks/muvera/prepare_data.py \
    --output_dir benchmarks/muvera/data --dataset nfcorpus --device cuda

# Run benchmark
PATH="benchmarks/muvera/.venv/bin:$PATH" python3 -u benchmarks/muvera/run_benchmark.py \
    --data_dir benchmarks/muvera/data --dataset nfcorpus --cleanup_first \
    --output benchmarks/muvera/benchmark_results.json
```
