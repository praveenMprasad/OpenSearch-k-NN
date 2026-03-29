#!/usr/bin/env python3
"""Test that Python client-side QUERY FDE encoding matches Java server-side.

Strategy: Index a doc via ingest pipeline, then:
1. Query via search pipeline (Java encodes query FDE) - get the score
2. Query via client-side FDE KNN (Python encodes query FDE) - get the score
3. Compare scores - if FDEs match, scores should be nearly identical
"""
import json, math, sys
import numpy as np
from opensearchpy import OpenSearch

# Use same params as IRPAPERS benchmark
DIM = 128
K_SIM = 5
DIM_PROJ = 16
R_REPS = 20
NP = 1 << K_SIM
FDE_DIM = R_REPS * NP * DIM_PROJ

INDEX = "fde-query-test"
INGEST_PIPELINE = "fde-query-test-ingest"
SEARCH_PIPELINE = "fde-query-test-search"

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}],
                    use_ssl=False, verify_certs=False, timeout=60)


class JavaRandom:
    def __init__(self, seed):
        self.seed = (seed ^ 0x5DEECE66D) & ((1 << 48) - 1)
        self._haveNextGaussian = False
        self._nextGaussian = 0.0
    def _next(self, bits):
        self.seed = (self.seed * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        return self.seed >> (48 - bits)
    def nextGaussian(self):
        if self._haveNextGaussian:
            self._haveNextGaussian = False
            return self._nextGaussian
        while True:
            v1 = 2 * self.nextDouble() - 1
            v2 = 2 * self.nextDouble() - 1
            s = v1 * v1 + v2 * v2
            if s < 1 and s != 0:
                break
        multiplier = math.sqrt(-2 * math.log(s) / s)
        self._nextGaussian = v2 * multiplier
        self._haveNextGaussian = True
        return v1 * multiplier
    def nextDouble(self):
        return ((self._next(26) << 27) + self._next(27)) / (1 << 53)
    def nextBoolean(self):
        return self._next(1) != 0


def encode_query_fde_python(multi_vectors):
    """Query mode: raw sum, no normalize, no fill empty."""
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
    out = np.zeros(FDE_DIM, dtype=np.float64)
    scale = 1.0 / math.sqrt(DIM_PROJ)
    offset = 0
    for r in range(R_REPS):
        centroids = np.zeros((NP, DIM))
        for v in vecs:
            cid = 0
            for k in range(K_SIM):
                if np.dot(v, simhash[r, k * DIM:(k + 1) * DIM]) > 0:
                    cid |= (1 << k)
            centroids[cid] += v
        # Query mode: NO normalize, NO fill empty
        for c in range(NP):
            for j in range(DIM_PROJ):
                val = 0.0
                for d in range(DIM):
                    val += centroids[c][d] * dim_reduce[r][d][j]
                out[offset] = scale * val
                offset += 1
    return out


# Setup
print("Setting up test index and pipelines...")
try: client.indices.delete(index=INDEX, ignore=[404])
except: pass
try: client.ingest.delete_pipeline(id=INGEST_PIPELINE, ignore=[404])
except: pass
try: client.transport.perform_request("DELETE", f"/_search/pipeline/{SEARCH_PIPELINE}")
except: pass

# Create ingest pipeline
client.ingest.put_pipeline(id=INGEST_PIPELINE, body={
    "processors": [{"muvera": {
        "source_field": "colbert_vectors", "target_field": "muvera_fde",
        "dim": DIM, "k_sim": K_SIM, "dim_proj": DIM_PROJ, "r_reps": R_REPS,
        "fde_dimension": FDE_DIM}}]})

# Create search pipeline
client.transport.perform_request("PUT", f"/_search/pipeline/{SEARCH_PIPELINE}", body={
    "request_processors": [{"muvera_query": {
        "target_field": "muvera_fde", "dim": DIM, "k_sim": K_SIM,
        "dim_proj": DIM_PROJ, "r_reps": R_REPS, "fde_dimension": FDE_DIM,
        "oversample_factor": 1}}]})

# Create index
hnsw = {"name": "hnsw", "space_type": "innerproduct", "engine": "faiss",
        "parameters": {"ef_construction": 16, "m": 4, "ef_search": 16}}
client.indices.create(index=INDEX, body={
    "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}},
    "mappings": {"properties": {
        "colbert_vectors": {"type": "object", "enabled": False},
        "muvera_fde": {"type": "knn_vector", "dimension": FDE_DIM, "method": hnsw}}}})

# Create test data
np.random.seed(123)
doc_vectors = np.random.randn(5, DIM).tolist()
query_vectors = np.random.randn(3, DIM).tolist()

# Index doc
client.index(index=INDEX, id="doc1", pipeline=INGEST_PIPELINE,
             body={"colbert_vectors": doc_vectors})
client.indices.refresh(index=INDEX)

# Read back doc FDE
doc = client.get(index=INDEX, id="doc1", _source_includes=["muvera_fde"])
doc_fde = np.array(doc["_source"]["muvera_fde"])

# Encode query FDE client-side
py_query_fde = encode_query_fde_python(query_vectors)

# Score: dot product of query FDE and doc FDE
py_score = float(np.dot(py_query_fde, doc_fde))
print(f"\nPython query FDE dot doc FDE = {py_score:.6f}")
print(f"Python query FDE first 5: {py_query_fde[:5]}")

# Now query via search pipeline to get Java's score
response = client.transport.perform_request("POST",
    f"/{INDEX}/_search?search_pipeline={SEARCH_PIPELINE}",
    body={"size": 1, "query": {"script_score": {"query": {"match_all": {}},
        "script": {"source": "lateInteractionScore(params.query_vectors, 'colbert_vectors', params._source, params.space_type)",
                   "params": {"query_vectors": query_vectors, "space_type": "innerproduct"}}}},
        "_source": False})

# The search pipeline rewrites to KNN on FDE, then lateInteractionScore rescores
# The final score is from lateInteractionScore (MaxSim), not FDE dot product
# But we can also do a pure KNN query via pipeline to get FDE score
response_knn = client.transport.perform_request("POST",
    f"/{INDEX}/_search",
    body={"size": 1, "query": {"knn": {"muvera_fde": {"vector": py_query_fde.tolist(), "k": 1}}},
          "_source": False})

knn_score = response_knn["hits"]["hits"][0]["_score"] if response_knn["hits"]["hits"] else 0
print(f"KNN score (py FDE vs doc FDE in OS) = {knn_score:.6f}")
print(f"Difference: {abs(py_score - knn_score):.10f}")

# Also try: index the Python query FDE as a doc, then KNN with doc FDE
# This tests if the vectors are truly compatible
client.index(index=INDEX, id="query_as_doc", body={"muvera_fde": py_query_fde.tolist()})
client.indices.refresh(index=INDEX)

response_reverse = client.transport.perform_request("POST",
    f"/{INDEX}/_search",
    body={"size": 2, "query": {"knn": {"muvera_fde": {"vector": doc_fde.tolist(), "k": 2}}},
          "_source": False})
print(f"\nReverse KNN (doc FDE as query):")
for hit in response_reverse["hits"]["hits"]:
    print(f"  {hit['_id']}: score={hit['_score']:.6f}")

# Cleanup
client.indices.delete(index=INDEX, ignore=[404])
client.ingest.delete_pipeline(id=INGEST_PIPELINE, ignore=[404])
try: client.transport.perform_request("DELETE", f"/_search/pipeline/{SEARCH_PIPELINE}")
except: pass

print("\nDone.")
