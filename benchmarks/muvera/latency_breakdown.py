#!/usr/bin/env python3
"""Single-query latency breakdown for MUVERA pipeline."""
import json, time, sys, os
import numpy as np
from opensearchpy import OpenSearch

INDEX_NAME = "muvera-benchmark-nfcorpus"
SEARCH_PIPELINE_OS1 = "muvera-search-benchmark-os1"
SEARCH_PIPELINE_OS4 = "muvera-search-benchmark-os4"
FDE_DIM = 10240

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}],
                    use_ssl=False, verify_certs=False, timeout=120)

# Load one query
data_dir = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/muvera/data"
with open(os.path.join(data_dir, "query_embeddings.json")) as f:
    query_data = json.load(f)
qid = sorted(query_data.keys())[0]
q_emb = query_data[qid]["embeddings"]
print(f"Query: {qid}, tokens: {len(q_emb)}, dim: {len(q_emb[0])}")

# Step 0: Warmup HNSW cache
print("\n--- Warmup ---")
try:
    r = client.transport.perform_request("GET", f"/_plugins/_knn/warmup/{INDEX_NAME}")
    print(f"KNN warmup: {r}")
except Exception as e:
    print(f"Warmup API: {e}")
# Also run a few dummy queries to warm caches
for _ in range(5):
    try:
        client.transport.perform_request("POST", f"/{INDEX_NAME}/_search",
            body={"size": 10, "query": {"knn": {"muvera_fde": {"vector": [0.0]*FDE_DIM, "k": 10}}}})
    except: pass
print("Warmup done")

# Step 1: Pure KNN on FDE (no search processor, precomputed FDE vector)
print("\n--- Step 1: Pure KNN on precomputed FDE vector (no processor overhead) ---")
# Use a dummy FDE vector just to measure ANN latency
dummy_fde = [0.01] * FDE_DIM
times = []
for _ in range(10):
    start = time.time()
    client.transport.perform_request("POST", f"/{INDEX_NAME}/_search",
        body={"size": 10, "query": {"knn": {"muvera_fde": {"vector": dummy_fde, "k": 10}}},
              "_source": False})
    times.append(time.time() - start)
print(f"  Pure KNN (10 runs): avg={np.mean(times)*1000:.1f}ms, p50={np.percentile(times,50)*1000:.1f}ms, p95={np.percentile(times,95)*1000:.1f}ms")

# Step 2: Pure KNN on mean vector
print("\n--- Step 2: Pure KNN on mean vector (128-dim) ---")
mean_q = np.array(q_emb).mean(axis=0).tolist()
times = []
for _ in range(10):
    start = time.time()
    client.transport.perform_request("POST", f"/{INDEX_NAME}/_search",
        body={"size": 10, "query": {"knn": {"mean_vector": {"vector": mean_q, "k": 10}}},
              "_source": False})
    times.append(time.time() - start)
print(f"  Pure KNN mean (10 runs): avg={np.mean(times)*1000:.1f}ms, p50={np.percentile(times,50)*1000:.1f}ms")

# Step 3: MUVERA-only via search processor (includes FDE encoding)
print("\n--- Step 3: MUVERA-only via search processor (encoding + KNN + lateInteraction) ---")
muvera_body = {"size": 10, "query": {"script_score": {"query": {"match_all": {}},
    "script": {"source": "lateInteractionScore(params.query_vectors, 'colbert_vectors', params._source, params.space_type)",
               "params": {"query_vectors": q_emb, "space_type": "innerproduct"}}}},
    "_source": {"excludes": ["muvera_fde", "mean_vector"]}}
times = []
for _ in range(5):
    start = time.time()
    client.transport.perform_request("POST",
        f"/{INDEX_NAME}/_search?search_pipeline={SEARCH_PIPELINE_OS1}", body=muvera_body)
    times.append(time.time() - start)
print(f"  MUVERA-only (5 runs): avg={np.mean(times)*1000:.1f}ms, p50={np.percentile(times,50)*1000:.1f}ms")

# Step 4: MUVERA + rerank 4x
print("\n--- Step 4: MUVERA + rerank 4x (encoding + KNN k=40 + lateInteraction on 40) ---")
times = []
for _ in range(5):
    start = time.time()
    client.transport.perform_request("POST",
        f"/{INDEX_NAME}/_search?search_pipeline={SEARCH_PIPELINE_OS4}", body=muvera_body)
    times.append(time.time() - start)
print(f"  MUVERA+rerank 4x (5 runs): avg={np.mean(times)*1000:.1f}ms, p50={np.percentile(times,50)*1000:.1f}ms")

# Step 5: Mean pool + rescore (no MUVERA processor)
print("\n--- Step 5: Mean pool + MaxSim rescore (KNN k=100 + lateInteraction on 100) ---")
rescore_body = {"size": 10,
    "query": {"knn": {"mean_vector": {"vector": mean_q, "k": 100}}},
    "rescore": {"query": {
        "rescore_query": {"script_score": {"query": {"match_all": {}},
            "script": {"source": "lateInteractionScore(params.query_vectors, 'colbert_vectors', params._source, params.space_type)",
                       "params": {"query_vectors": q_emb, "space_type": "innerproduct"}}}},
        "query_weight": 0, "rescore_query_weight": 1}},
    "_source": {"excludes": ["muvera_fde", "mean_vector"]}}
times = []
for _ in range(5):
    start = time.time()
    client.transport.perform_request("POST", f"/{INDEX_NAME}/_search", body=rescore_body)
    times.append(time.time() - start)
print(f"  Mean+rescore (5 runs): avg={np.mean(times)*1000:.1f}ms, p50={np.percentile(times,50)*1000:.1f}ms")

# Step 6: Precomputed FDE + KNN + lateInteraction (no search processor)
# This is what Qdrant measures: client encodes FDE, DB does KNN + rerank
print("\n--- Step 6: Precomputed FDE + KNN + lateInteraction (Qdrant-comparable) ---")
# Encode FDE client-side using same algorithm as Python test
# Encode FDE client-side using same algorithm
DIM,K_SIM,DIM_PROJ,R_REPS=128,5,16,20
NP=1<<K_SIM
rng_enc=np.random.RandomState(42)
sh=rng_enc.randn(R_REPS,K_SIM*DIM)
dr=np.where(rng_enc.randint(0,2,size=(R_REPS,DIM,DIM_PROJ))==1,1.0,-1.0)

def encode_query_fde(vecs):
    out=np.zeros(R_REPS*NP*DIM_PROJ,dtype=np.float32)
    s=1/np.sqrt(DIM_PROJ); o=0
    for r in range(R_REPS):
        c=np.zeros((NP,DIM))
        for v in vecs:
            cid=0
            for k in range(K_SIM):
                if np.dot(v,sh[r,k*DIM:(k+1)*DIM])>0: cid|=(1<<k)
            c[cid]+=v
        for ci in range(NP): out[o:o+DIM_PROJ]=s*(c[ci]@dr[r]); o+=DIM_PROJ
    return out

q_arr = np.array(q_emb)
precomputed_fde = encode_query_fde(q_arr).tolist()

# 6a: KNN on precomputed FDE only (MUVERA-only equivalent, no processor)
print("  6a: Precomputed FDE KNN only (k=10, no lateInteraction):")
times = []
for _ in range(10):
    start = time.time()
    client.transport.perform_request("POST", f"/{INDEX_NAME}/_search",
        body={"size": 10, "query": {"knn": {"muvera_fde": {"vector": precomputed_fde, "k": 10}}},
              "_source": False})
    times.append(time.time() - start)
print(f"      avg={np.mean(times)*1000:.1f}ms, p50={np.percentile(times,50)*1000:.1f}ms")

# 6b: KNN on precomputed FDE + lateInteraction rerank (k=10)
print("  6b: Precomputed FDE KNN (k=10) + lateInteraction on 10 docs:")
times = []
for _ in range(5):
    start = time.time()
    client.transport.perform_request("POST", f"/{INDEX_NAME}/_search",
        body={"size": 10, "query": {"script_score": {
                "query": {"knn": {"muvera_fde": {"vector": precomputed_fde, "k": 10}}},
                "script": {"source": "lateInteractionScore(params.query_vectors, 'colbert_vectors', params._source, params.space_type)",
                           "params": {"query_vectors": q_emb, "space_type": "innerproduct"}}}},
              "_source": {"excludes": ["muvera_fde", "mean_vector"]}})
    times.append(time.time() - start)
print(f"      avg={np.mean(times)*1000:.1f}ms, p50={np.percentile(times,50)*1000:.1f}ms")

# 6c: KNN on precomputed FDE + lateInteraction rerank (k=40, 4x oversample)
print("  6c: Precomputed FDE KNN (k=40) + lateInteraction on 40 docs:")
times = []
for _ in range(5):
    start = time.time()
    client.transport.perform_request("POST", f"/{INDEX_NAME}/_search",
        body={"size": 10, "query": {"script_score": {
                "query": {"knn": {"muvera_fde": {"vector": precomputed_fde, "k": 40}}},
                "script": {"source": "lateInteractionScore(params.query_vectors, 'colbert_vectors', params._source, params.space_type)",
                           "params": {"query_vectors": q_emb, "space_type": "innerproduct"}}}},
              "_source": {"excludes": ["muvera_fde", "mean_vector"]}})
    times.append(time.time() - start)
print(f"      avg={np.mean(times)*1000:.1f}ms, p50={np.percentile(times,50)*1000:.1f}ms")

print("\n--- Summary ---")
print("Pure KNN on FDE = ANN search cost only (no encoding, no scoring)")
print("MUVERA-only - Pure KNN = FDE encoding + lateInteraction on 10 docs")
print("MUVERA+rerank - MUVERA-only = extra lateInteraction on 30 more docs")
print("\nQdrant-comparable (Step 6): client-side FDE encoding, server does KNN + rerank only")
print("  Qdrant reported: MUVERA-only 150ms, MUVERA+rerank 180ms")
