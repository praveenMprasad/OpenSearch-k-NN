# MUVERA Benchmark Results

## Overview

This benchmark evaluates the MUVERA (Multi-Vector Retrieval via Fixed Dimensional Encodings) processors
for OpenSearch on the BeIR nfcorpus dataset using ColBERTv2 multi-vector embeddings.

MUVERA converts variable-length multi-vector representations into fixed-size single vectors (FDE) that
approximate MaxSim scoring via dot product. This enables fast ANN retrieval on standard knn_vector fields
while preserving multi-vector retrieval quality through optional MaxSim reranking.

## Dataset

- **BeIR nfcorpus**: 3,633 documents, 323 test queries with human-annotated relevance judgments
- **Embedding model**: ColBERTv2 (128-dimensional token vectors)
- **Metric**: NDCG@1, NDCG@5, NDCG@10 against BeIR qrels

## MUVERA Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| dim | 128 | ColBERTv2 token vector dimension |
| k_sim | 5 | SimHash hyperplanes (2^5 = 32 clusters) |
| dim_proj | 16 | Projected dimension per cluster |
| r_reps | 20 | Independent repetitions |
| FDE dimension | 10,240 | Output: 20 x 32 x 16 |

## Index Configuration

- Engine: faiss
- Algorithm: HNSW
- Space type: innerproduct
- m: 16, ef_construction: 512, ef_search: 512
- 1 shard, 0 replicas, force merged to 1 segment
- HNSW graphs warmed up before queries

## Approaches Evaluated

1. **Exact MaxSim (brute-force)**: Offline computation of MaxSim between all query-document pairs.
   Represents the theoretical quality ceiling for multi-vector retrieval. No approximation.

2. **Mean pool + MaxSim rerank**: Average all token vectors into a single 128-dim vector, perform ANN
   search (k=100), then rescore top candidates with lateInteractionScore. This is the naive baseline
   that users would employ without MUVERA.

3. **MUVERA-only**: Encode query multi-vectors into a 10,240-dim FDE via the MUVERA search processor,
   perform ANN search on the FDE field (k=10). The lateInteractionScore still runs on the returned
   candidates but only rescores the same 10 results.

4. **MUVERA + MaxSim rerank (4x)**: Same as MUVERA-only but with 4x oversampling. ANN fetches 40
   candidates, lateInteractionScore rescores all 40, returns top 10.

## Results

| Approach | NDCG@1 | NDCG@5 | NDCG@10 | Avg Latency (ms) | P95 Latency (ms) |
|----------|--------|--------|---------|-------------------|-------------------|
| Exact MaxSim (brute-force) | 0.483 | 0.386 | 0.344 | offline | - |
| MUVERA + MaxSim rerank (4x) | 0.474 | 0.379 | 0.339 | 18,205 | 18,934 |
| MUVERA-only | 0.464 | 0.369 | 0.329 | 5,498 | 5,856 |
| Mean pool + MaxSim rerank | 0.249 | 0.183 | 0.145 | 550 | 665 |

## Key Findings

**MUVERA + rerank recovers 98.5% of exact MaxSim quality** (NDCG@10: 0.339 vs 0.344), demonstrating
that the FDE approximation combined with MaxSim reranking provides near-lossless multi-vector retrieval.

**MUVERA-only achieves 95.6% of exact MaxSim quality** (NDCG@10: 0.329 vs 0.344) without any reranking,
showing that the FDE encoding alone captures most of the multi-vector signal.

**Mean pooling loses most of the multi-vector quality** (NDCG@10: 0.145 vs 0.344, only 42% of the
ceiling). This demonstrates why MUVERA is needed: naive single-vector approaches discard the fine-grained
token-level information that makes multi-vector models effective.

## Latency Notes

The latency numbers reflect an unoptimized single-node development setup and are not representative of
production performance. The primary bottleneck is the per-query FDE encoding in the MUVERA search
processor (Java, single-threaded). In a production deployment, query FDE encoding could be optimized
through SIMD vectorization, precomputation, or caching. The quality metrics (NDCG) are the primary
focus of this benchmark.

## Reproducibility

To reproduce these results:

```bash
# 1. Build and start OpenSearch with the k-NN plugin
./gradlew assemble -x test -x integTest
./gradlew run -x test -x integTest &

# 2. Set up Python environment
python3 -m venv benchmarks/muvera/.venv
benchmarks/muvera/.venv/bin/pip install -r benchmarks/muvera/requirements.txt

# 3. Prepare data (encode nfcorpus with ColBERTv2)
PATH="benchmarks/muvera/.venv/bin:$PATH" python3 benchmarks/muvera/prepare_data.py \
    --output_dir benchmarks/muvera/data --device cuda

# 4. Run benchmark
PATH="benchmarks/muvera/.venv/bin:$PATH" python3 -u benchmarks/muvera/run_benchmark.py \
    --data_dir benchmarks/muvera/data --cleanup_first \
    --output benchmarks/muvera/benchmark_results.json
```
