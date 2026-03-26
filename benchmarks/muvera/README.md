# MUVERA Benchmark

Benchmarks MUVERA multi-vector retrieval on the BeIR nfcorpus dataset with ColBERTv2,
comparable to [Qdrant's MUVERA benchmark](https://qdrant.tech/articles/muvera-embeddings/).

## Setup

```bash
pip install -r requirements.txt
```

## Step 1: Prepare Data

Downloads nfcorpus, encodes with ColBERTv2, computes brute-force MaxSim ground truth.

```bash
python prepare_data.py --output_dir ./data --device cpu
# Use --device mps on Apple Silicon for faster encoding
# Use --device cuda if you have a GPU
```

This takes ~10-20 minutes on CPU (3,633 docs + 323 queries).

## Step 2: Start OpenSearch

Build and run OpenSearch with the k-NN plugin from your local branch:

```bash
# From the k-NN repo root
./gradlew assemble
# Then start OpenSearch with the plugin installed
```

Or use `./gradlew run` from the OpenSearch root if wired up.

## Step 3: Run Benchmark

```bash
python run_benchmark.py --data_dir ./data --host localhost --port 9200
```

Options:
- `--size 10` — number of results (default 10)
- `--cleanup_first` — delete existing index/pipelines before running
- `--output ./results.json` — save results to file

## What It Measures

| Approach | Description |
|----------|-------------|
| MUVERA-only | ANN search on FDE vectors, no reranking |
| MUVERA + rerank (2x/4x/10x) | ANN with oversampling, then MaxSim reranking via lateInteractionScore |

Metrics: NDCG@1, @5, @10 (vs brute-force MaxSim ground truth), search latency percentiles.

MUVERA params: `k_sim=5, dim_proj=16, r_reps=20` (FDE dim=10240), matching Qdrant's config.
