#!/usr/bin/env python3
"""
MUVERA Benchmark Runner for OpenSearch - IRPAPERS + ColModernVBERT.

Key differences from run_benchmark.py (nfcorpus/scifact):
- Uses Recall@K (not NDCG) since IRPAPERS has single-relevance per query
- Uses Weaviate's MUVERA params: k_sim=4, dim_proj=16, r_reps=10 -> FDE dim 2,560
- Compares against Weaviate's published MUVERA results (Table 3 in IRPAPERS paper)
- ColModernVBERT produces ~1000 128-dim vectors per page (vs ~30 for ColBERTv2)

Usage:
    python run_benchmark_irpapers.py --data_dir ./data_irpapers --cleanup_first
"""
import argparse, json, math, os, time
import numpy as np
from opensearchpy import OpenSearch, helpers
from tqdm import tqdm

# Match Weaviate's MUVERA params from the IRPAPERS paper
MUVERA_PARAMS = {"dim": 128, "k_sim": 4, "dim_proj": 16, "r_reps": 10}
FDE_DIM = MUVERA_PARAMS["r_reps"] * (2 ** MUVERA_PARAMS["k_sim"]) * MUVERA_PARAMS["dim_proj"]
# 10 * 16 * 16 = 2560

INDEX_NAME = "muvera-benchmark-irpapers"
INGEST_PIPELINE = "muvera-ingest-irpapers"
SEARCH_PIPELINE = "muvera-search-irpapers"


def create_client(host, port, username, password):
    auth = (username, password) if username else None
    return OpenSearch(hosts=[{"host": host, "port": port}], http_auth=auth,
                      use_ssl=False, verify_certs=False, timeout=300)


def compute_recall(ranked_ids, relevant_ids, k):
    """Recall@K for single-relevance: 1 if relevant doc in top-K, else 0."""
    return 1.0 if any(did in relevant_ids for did in ranked_ids[:k]) else 0.0



def eval_bruteforce_maxsim(ground_truth, qrels, k_values=(1, 5, 10, 20)):
    """Evaluate exact brute-force MaxSim (offline ground truth)."""
    print("\n--- Evaluating: Exact brute-force MaxSim (offline) ---")
    recall_scores = {k: [] for k in k_values}
    for qid, rankings in tqdm(ground_truth.items(), desc="Brute-force MaxSim"):
        qrel = qrels.get(qid, {})
        if not qrel:
            continue
        relevant_ids = set(qrel.keys())
        ranked_ids = [e["doc_id"] for e in rankings]
        for k in k_values:
            recall_scores[k].append(compute_recall(ranked_ids, relevant_ids, k))
    results = {"config": "Exact MaxSim (brute-force)", "num_queries": len(recall_scores[k_values[0]])}
    for k in k_values:
        results[f"recall@{k}"] = np.mean(recall_scores[k]) if recall_scores[k] else 0.0
    results.update({"avg_latency_ms": 0, "p50_latency_ms": 0, "p95_latency_ms": 0})
    return results


def setup_ingest_pipeline(client):
    client.ingest.put_pipeline(id=INGEST_PIPELINE, body={
        "description": "MUVERA IRPAPERS ingest pipeline",
        "processors": [{"muvera": {
            "source_field": "colbert_vectors", "target_field": "muvera_fde",
            "dim": MUVERA_PARAMS["dim"], "k_sim": MUVERA_PARAMS["k_sim"],
            "dim_proj": MUVERA_PARAMS["dim_proj"], "r_reps": MUVERA_PARAMS["r_reps"],
            "fde_dimension": FDE_DIM}}]})
    print(f"Created ingest pipeline: {INGEST_PIPELINE} (FDE dim={FDE_DIM})")


def setup_search_pipeline(client, oversample_factor):
    name = f"{SEARCH_PIPELINE}-os{oversample_factor}"
    client.transport.perform_request("PUT", f"/_search/pipeline/{name}", body={
        "description": f"MUVERA IRPAPERS search pipeline (oversample={oversample_factor})",
        "request_processors": [{"muvera_query": {
            "target_field": "muvera_fde", "dim": MUVERA_PARAMS["dim"],
            "k_sim": MUVERA_PARAMS["k_sim"], "dim_proj": MUVERA_PARAMS["dim_proj"],
            "r_reps": MUVERA_PARAMS["r_reps"], "fde_dimension": FDE_DIM,
            "oversample_factor": oversample_factor}}]})
    print(f"Created search pipeline: {name}")


def create_index(client, ef_search=512):
    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)
    hnsw_params = {"name": "hnsw", "space_type": "innerproduct", "engine": "faiss",
                   "parameters": {"ef_construction": 512, "m": 16, "ef_search": ef_search}}
    client.indices.create(index=INDEX_NAME, body={
        "settings": {"index": {"knn": True, "number_of_shards": 1,
                               "number_of_replicas": 0, "refresh_interval": "-1"}},
        "mappings": {"properties": {
            "doc_id": {"type": "keyword"}, "text": {"type": "text"},
            "colbert_vectors": {"type": "object", "enabled": False},
            "mean_vector": {"type": "knn_vector", "dimension": 128, "method": hnsw_params},
            "muvera_fde": {"type": "knn_vector", "dimension": FDE_DIM, "method": hnsw_params}}}})
    print(f"Created index: {INDEX_NAME} (1 shard, FDE dim={FDE_DIM})")


def index_documents(client, doc_data):
    print(f"Indexing {len(doc_data)} documents...")
    actions = []
    for did, data in doc_data.items():
        mean_vec = np.array(data["embeddings"]).mean(axis=0).tolist()
        actions.append({"_index": INDEX_NAME, "_id": did, "pipeline": INGEST_PIPELINE,
                        "_source": {"doc_id": did, "text": data.get("text", ""),
                                    "colbert_vectors": data["embeddings"],
                                    "mean_vector": mean_vec}})
    start = time.time()
    try:
        success, errors = helpers.bulk(client, actions, chunk_size=20, request_timeout=1200)
    except Exception as e:
        err_msg = str(type(e).__name__)
        if hasattr(e, "errors") and e.errors:
            sample = e.errors[0] if e.errors else {}
            if isinstance(sample, dict):
                for action_type in ["index", "create", "update"]:
                    if action_type in sample:
                        err_info = sample[action_type].get("error", {})
                        err_msg = f"{err_info.get('type', 'unknown')}: {err_info.get('reason', 'unknown')[:200]}"
                        break
            print(f"Bulk indexing failed: {err_msg} ({len(e.errors)} errors)")
        else:
            print(f"Bulk indexing failed: {err_msg}")
        raise
    elapsed = time.time() - start
    print(f"Indexed {success} docs in {elapsed:.1f}s ({success / elapsed:.1f} docs/sec)")
    if errors:
        print(f"  Errors: {len(errors)}")
    client.indices.refresh(index=INDEX_NAME)
    print("Refreshed index")
    print("Force merging to 1 segment...")
    client.indices.forcemerge(index=INDEX_NAME, max_num_segments=1)
    print("Force merge complete")
    return elapsed



def warmup_cache(client):
    print("Warming up HNSW cache...")
    try:
        client.transport.perform_request("GET", f"/_plugins/_knn/warmup/{INDEX_NAME}")
        print("KNN warmup complete")
    except Exception as e:
        print(f"KNN warmup API: {e}, using search warmup")
        dummy = [0.0] * FDE_DIM
        for _ in range(3):
            try:
                client.transport.perform_request("POST", f"/{INDEX_NAME}/_search",
                    body={"size": 10, "query": {"knn": {"muvera_fde": {"vector": dummy, "k": 10}}}})
            except Exception:
                pass


def build_muvera_query(query_embeddings, size=20):
    """Build MUVERA query with lateInteractionScore reranking."""
    return {"size": size, "query": {"script_score": {"query": {"match_all": {}},
        "script": {"source": "lateInteractionScore(params.query_vectors, 'colbert_vectors', params._source, params.space_type)",
                   "params": {"query_vectors": query_embeddings, "space_type": "innerproduct"}}}},
        "_source": {"excludes": ["muvera_fde", "mean_vector"]}}


def build_mean_pool_rerank_query(query_embeddings, size=20, prefetch_k=100):
    mean_query = np.array(query_embeddings).mean(axis=0).tolist()
    return {"size": size,
        "query": {"knn": {"mean_vector": {"vector": mean_query, "k": prefetch_k}}},
        "rescore": {"query": {
            "rescore_query": {"script_score": {"query": {"match_all": {}},
                "script": {"source": "lateInteractionScore(params.query_vectors, 'colbert_vectors', params._source, params.space_type)",
                           "params": {"query_vectors": query_embeddings, "space_type": "innerproduct"}}}},
            "query_weight": 0, "rescore_query_weight": 1}},
        "_source": {"excludes": ["muvera_fde", "mean_vector"]}}


def run_muvera_query(client, query_embeddings, size, oversample_factor):
    name = f"{SEARCH_PIPELINE}-os{oversample_factor}"
    return client.transport.perform_request("POST",
        f"/{INDEX_NAME}/_search?search_pipeline={name}",
        body=build_muvera_query(query_embeddings, size))


def run_mean_pool_rerank_query(client, query_embeddings, size, prefetch_k=100):
    return client.transport.perform_request("POST", f"/{INDEX_NAME}/_search",
        body=build_mean_pool_rerank_query(query_embeddings, size, prefetch_k))


def extract_ranked_list(response):
    return [(hit["_id"], hit["_score"]) for hit in response.get("hits", {}).get("hits", [])]


def run_bm25_query(client, query_text, size=20):
    """Run BM25 text search on the transcription field."""
    return client.transport.perform_request("POST", f"/{INDEX_NAME}/_search",
        body={"size": size, "query": {"match": {"text": query_text}},
              "_source": False})


def encode_query_fde_client(multi_vectors):
    """Encode query multi-vectors into FDE client-side (pure Python).
    Matches MUVERA params: k_sim=4, dim_proj=16, r_reps=10, dim=128.
    Uses Java-compatible Random (LCG) with seed 42 to match server-side encoder.
    """
    DIM = MUVERA_PARAMS["dim"]
    K_SIM = MUVERA_PARAMS["k_sim"]
    DIM_PROJ = MUVERA_PARAMS["dim_proj"]
    R_REPS = MUVERA_PARAMS["r_reps"]
    NP = 1 << K_SIM

    class JavaRandom:
        def __init__(self, seed):
            self.seed = (seed ^ 0x5DEECE66D) & ((1 << 48) - 1)
        def _next(self, bits):
            self.seed = (self.seed * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
            return self.seed >> (48 - bits)
        def nextGaussian(self):
            import math
            while True:
                v1 = 2 * self.nextDouble() - 1
                v2 = 2 * self.nextDouble() - 1
                s = v1 * v1 + v2 * v2
                if s < 1 and s != 0:
                    break
            multiplier = math.sqrt(-2 * math.log(s) / s)
            return v1 * multiplier
        def nextDouble(self):
            return ((self._next(26) << 27) + self._next(27)) / (1 << 53)
        def nextBoolean(self):
            return self._next(1) != 0

    rng = JavaRandom(42)

    simhash = np.zeros((R_REPS, K_SIM * DIM))
    for r in range(R_REPS):
        for i in range(K_SIM * DIM):
            simhash[r][i] = rng.nextGaussian()

    dim_reduce = np.zeros((R_REPS, DIM, DIM_PROJ))
    for r in range(R_REPS):
        for i in range(DIM):
            for j in range(DIM_PROJ):
                dim_reduce[r][i][j] = 1.0 if rng.nextBoolean() else -1.0

    vecs = np.array(multi_vectors)
    out = np.zeros(R_REPS * NP * DIM_PROJ, dtype=np.float32)
    scale = 1.0 / np.sqrt(DIM_PROJ)
    offset = 0
    for r in range(R_REPS):
        centroids = np.zeros((NP, DIM))
        for v in vecs:
            cid = 0
            for k in range(K_SIM):
                if np.dot(v, simhash[r, k * DIM:(k + 1) * DIM]) > 0:
                    cid |= (1 << k)
            centroids[cid] += v
        for ci in range(NP):
            out[offset:offset + DIM_PROJ] = scale * (centroids[ci] @ dim_reduce[r])
            offset += DIM_PROJ
    return out.tolist()


def run_pure_fde_query(client, query_embeddings, size=20):
    """Pure FDE ANN query — no lateInteractionScore, no search pipeline."""
    fde = encode_query_fde_client(query_embeddings)
    return client.transport.perform_request("POST", f"/{INDEX_NAME}/_search",
        body={"size": size,
              "query": {"knn": {"muvera_fde": {"vector": fde, "k": size}}},
              "_source": False})


def run_os_benchmark(client, config_name, query_fn, query_data, qrels,
                     size=20, k_values=(1, 5, 10, 20)):
    """Run benchmark and compute Recall@K."""
    print(f"\n--- Running: {config_name} ---")
    recall_scores = {k: [] for k in k_values}
    latencies = []
    for qid in tqdm(sorted(query_data.keys()), desc=config_name):
        qrel = qrels.get(qid, {})
        if not qrel:
            continue
        relevant_ids = set(qrel.keys())
        start = time.time()
        try:
            response = query_fn(client, query_data[qid]["embeddings"], size)
            elapsed = time.time() - start
            latencies.append(elapsed)
            ranked_ids = [did for did, _ in extract_ranked_list(response)]
            for k in k_values:
                recall_scores[k].append(compute_recall(ranked_ids, relevant_ids, k))
        except Exception as e:
            msg = str(e)[:200]
            print(f"  Query {qid} failed: {msg}")
            latencies.append(time.time() - start)
    results = {"config": config_name, "num_queries": len(latencies)}
    for k in k_values:
        results[f"recall@{k}"] = np.mean(recall_scores[k]) if recall_scores[k] else 0.0
    results["avg_latency_ms"] = np.mean(latencies) * 1000 if latencies else 0.0
    results["p50_latency_ms"] = np.percentile(latencies, 50) * 1000 if latencies else 0.0
    results["p95_latency_ms"] = np.percentile(latencies, 95) * 1000 if latencies else 0.0
    return results


def run_text_benchmark(client, config_name, query_fn, query_data, qrels,
                       size=20, k_values=(1, 5, 10, 20)):
    """Run benchmark using query text (not embeddings) — for BM25."""
    print(f"\n--- Running: {config_name} ---")
    recall_scores = {k: [] for k in k_values}
    latencies = []
    for qid in tqdm(sorted(query_data.keys()), desc=config_name):
        qrel = qrels.get(qid, {})
        if not qrel:
            continue
        relevant_ids = set(qrel.keys())
        start = time.time()
        try:
            response = query_fn(client, query_data[qid]["text"], size)
            elapsed = time.time() - start
            latencies.append(elapsed)
            ranked_ids = [did for did, _ in extract_ranked_list(response)]
            for k in k_values:
                recall_scores[k].append(compute_recall(ranked_ids, relevant_ids, k))
        except Exception as e:
            msg = str(e)[:200]
            print(f"  Query {qid} failed: {msg}")
            latencies.append(time.time() - start)
    results = {"config": config_name, "num_queries": len(latencies)}
    for k in k_values:
        results[f"recall@{k}"] = np.mean(recall_scores[k]) if recall_scores[k] else 0.0
    results["avg_latency_ms"] = np.mean(latencies) * 1000 if latencies else 0.0
    results["p50_latency_ms"] = np.percentile(latencies, 50) * 1000 if latencies else 0.0
    results["p95_latency_ms"] = np.percentile(latencies, 95) * 1000 if latencies else 0.0
    return results



def print_results(all_results):
    print("\n" + "=" * 110)
    print("MUVERA Benchmark Results - IRPAPERS + ColModernVBERT")
    print(f"MUVERA params: k_sim={MUVERA_PARAMS['k_sim']}, dim_proj={MUVERA_PARAMS['dim_proj']}, "
          f"r_reps={MUVERA_PARAMS['r_reps']}, FDE dim={FDE_DIM}")
    print(f"HNSW: m=16, ef_construction=512")
    print("=" * 110)
    hdr = (f"{'Approach':<40} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@20':>8} "
           f"{'Avg(ms)':>10} {'P95(ms)':>10}")
    print(hdr)
    print("-" * 110)
    for r in all_results:
        lat = f"{r['avg_latency_ms']:>10.1f}" if r['avg_latency_ms'] > 0 else "   offline"
        p95 = f"{r['p95_latency_ms']:>10.1f}" if r['p95_latency_ms'] > 0 else "         -"
        r1 = f"{r.get('recall@1', 0):>8.1%}"
        r5 = f"{r.get('recall@5', 0):>8.1%}"
        r10 = f"{r.get('recall@10', 0):>8.1%}"
        r20 = f"{r.get('recall@20', 0):>8.1%}"
        print(f"{r['config']:<40} {r1} {r5} {r10} {r20} {lat} {p95}")
    print("=" * 110)

    # Weaviate reference numbers from IRPAPERS paper
    print("\nWeaviate reference (IRPAPERS paper):")
    print(f"{'Approach':<40} {'R@1':>8} {'R@5':>8} {'R@20':>8}")
    print("-" * 70)
    print(f"{'ColModernVBERT (exact, Weaviate)':<40} {'43%':>8} {'78%':>8} {'93%':>8}")
    print(f"{'ColModernVBERT+MUVERA ef=1024 (Weaviate)':<40} {'41%':>8} {'75%':>8} {'88%':>8}")
    print(f"{'ColModernVBERT+MUVERA ef=512 (Weaviate)':<40} {'37%':>8} {'68%':>8} {'78%':>8}")
    print(f"{'ColModernVBERT+MUVERA ef=256 (Weaviate)':<40} {'35%':>8} {'61%':>8} {'66%':>8}")
    print()
    print("Other Weaviate baselines (for context):")
    print(f"{'Hybrid Text (Arctic 2.0 + BM25)':<40} {'46%':>8} {'78%':>8} {'91%':>8}")
    print(f"{'Multimodal Hybrid (text+image)':<40} {'49%':>8} {'81%':>8} {'95%':>8}")
    print(f"{'ColPali (2.9B params)':<40} {'45%':>8} {'79%':>8} {'93%':>8}")
    print(f"{'ColQwen2 (2.2B params)':<40} {'49%':>8} {'81%':>8} {'94%':>8}")
    print(f"{'Cohere Embed v4 (closed-source)':<40} {'58%':>8} {'87%':>8} {'97%':>8}")


def cleanup(client):
    try:
        client.indices.delete(index=INDEX_NAME, ignore=[404])
    except Exception:
        pass
    try:
        client.ingest.delete_pipeline(id=INGEST_PIPELINE, ignore=[404])
    except Exception:
        pass
    for osf in [1, 2, 4, 8]:
        try:
            client.transport.perform_request(
                "DELETE", f"/_search/pipeline/{SEARCH_PIPELINE}-os{osf}")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Run MUVERA IRPAPERS benchmark on OpenSearch")
    parser.add_argument("--data_dir", type=str, default="./data_irpapers")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--username", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--size", type=int, default=20,
                        help="Number of results to return (need >=20 for Recall@20)")
    parser.add_argument("--ef_search", type=int, default=512,
                        help="HNSW ef_search parameter")
    parser.add_argument("--skip_setup", action="store_true")
    parser.add_argument("--cleanup_first", action="store_true")
    parser.add_argument("--output", type=str, default="./benchmark_results_irpapers.json")
    args = parser.parse_args()

    print("Loading prepared IRPAPERS data...")
    with open(os.path.join(args.data_dir, "doc_embeddings.json")) as f:
        doc_data = json.load(f)
    with open(os.path.join(args.data_dir, "query_embeddings.json")) as f:
        query_data = json.load(f)
    with open(os.path.join(args.data_dir, "qrels.json")) as f:
        qrels = json.load(f)
    with open(os.path.join(args.data_dir, "ground_truth_maxsim.json")) as f:
        ground_truth = json.load(f)
    print(f"Loaded {len(doc_data)} pages, {len(query_data)} queries, {len(qrels)} qrels")

    # Verify data looks right
    sample_doc = next(iter(doc_data.values()))
    sample_query = next(iter(query_data.values()))
    print(f"  Sample doc: {len(sample_doc['embeddings'])} tokens x {len(sample_doc['embeddings'][0])} dim")
    print(f"  Sample query: {len(sample_query['embeddings'])} tokens x {len(sample_query['embeddings'][0])} dim")

    all_results = []
    all_results.append(eval_bruteforce_maxsim(ground_truth, qrels))

    client = create_client(args.host, args.port, args.username, args.password)
    info = client.info()
    print(f"Connected to OpenSearch {info['version']['number']}")

    if args.cleanup_first:
        cleanup(client)

    if args.skip_setup:
        client.indices.put_settings(index=INDEX_NAME,
            body={"index.knn.algo_param.ef_search": args.ef_search})
        print(f"Set ef_search={args.ef_search} on existing index")
        warmup_cache(client)
        index_time = 0
    else:
        setup_ingest_pipeline(client)
        create_index(client, ef_search=args.ef_search)
        for osf in [1, 2, 4, 8]:
            setup_search_pipeline(client, osf)
        index_time = index_documents(client, doc_data)
        client.indices.put_settings(index=INDEX_NAME,
            body={"index.knn.algo_param.ef_search": args.ef_search})
        print(f"Set ef_search={args.ef_search}")
        warmup_cache(client)

    # Config 1: BM25 text search
    all_results.append(run_text_benchmark(client, "BM25 (text only)",
        lambda c, text, sz: run_bm25_query(c, text, sz),
        query_data, qrels, size=args.size))

    # Config 2: Pure MUVERA FDE-only (client-side encoding, no MaxSim rerank)
    all_results.append(run_os_benchmark(client, "MUVERA FDE-only (no rerank)",
        lambda c, emb, sz: run_pure_fde_query(c, emb, sz),
        query_data, qrels, size=args.size))

    # Config 3: MUVERA + MaxSim rerank (4x oversample)
    all_results.append(run_os_benchmark(client, "MUVERA + rerank (4x)",
        lambda c, emb, sz: run_muvera_query(c, emb, sz, oversample_factor=4),
        query_data, qrels, size=args.size))

    # Config 5: MUVERA + MaxSim rerank (8x oversample)
    all_results.append(run_os_benchmark(client, "MUVERA + rerank (8x)",
        lambda c, emb, sz: run_muvera_query(c, emb, sz, oversample_factor=8),
        query_data, qrels, size=args.size))

    print_results(all_results)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    stats = client.indices.stats(index=INDEX_NAME)
    store_size = stats["indices"][INDEX_NAME]["total"]["store"]["size_in_bytes"]
    print(f"Index size: {store_size / (1024 * 1024):.1f} MB, Ingest time: {index_time:.1f}s")


if __name__ == "__main__":
    main()
