#!/usr/bin/env python3
"""Test that Python client-side FDE encoding matches Java server-side encoding.

Indexes a test doc with known vectors via MUVERA ingest pipeline,
reads back the muvera_fde field, and compares with Python-encoded FDE.
"""
import json, math, sys
import numpy as np
from opensearchpy import OpenSearch

# MUVERA params for nfcorpus (k_sim=5, dim_proj=16, r_reps=20)
DIM = 128
K_SIM = 5
DIM_PROJ = 16
R_REPS = 20
NP = 1 << K_SIM  # 32
FDE_DIM = R_REPS * NP * DIM_PROJ  # 10240

INDEX = "muvera-fde-test"
PIPELINE = "muvera-ingest-benchmark"

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


def encode_document_fde_python(multi_vectors):
    """Python document FDE encoding matching Java's processDocument()."""
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
    out = np.zeros(R_REPS * NP * DIM_PROJ, dtype=np.float64)
    scale = 1.0 / math.sqrt(DIM_PROJ)
    offset = 0

    for r in range(R_REPS):
        centers = np.zeros((NP, DIM))
        counts = np.zeros(NP, dtype=int)
        cluster_vec_indices = [[] for _ in range(NP)]

        for vi, v in enumerate(vecs):
            cid = 0
            for k in range(K_SIM):
                dot = np.dot(v, simhash[r, k * DIM:(k + 1) * DIM])
                if dot > 0:
                    cid |= (1 << k)
            centers[cid] += v
            counts[cid] += 1
            cluster_vec_indices[cid].append(vi)

        # Normalize by count (document mode)
        for c in range(NP):
            if counts[c] > 0:
                centers[c] /= counts[c]

        # Fill empty clusters from Hamming-nearest non-empty
        for c in range(NP):
            if counts[c] == 0:
                nearest = -1
                min_dist = 999
                for other in range(NP):
                    if counts[other] > 0:
                        dist = bin(c ^ other).count('1')
                        if dist < min_dist:
                            min_dist = dist
                            nearest = other
                if nearest >= 0:
                    vec_idx = cluster_vec_indices[nearest][0]
                    centers[c] = vecs[vec_idx].copy()

        # Project
        for c in range(NP):
            for j in range(DIM_PROJ):
                val = 0.0
                for d in range(DIM):
                    val += centers[c][d] * dim_reduce[r][d][j]
                out[offset] = scale * val
                offset += 1

    return out


# Create a simple test vector set
np.random.seed(123)
test_vectors = np.random.randn(3, DIM).tolist()  # 3 vectors of dim 128

print(f"Test vectors: {len(test_vectors)} x {len(test_vectors[0])}")

# Step 1: Encode with Python
py_fde = encode_document_fde_python(test_vectors)
print(f"Python FDE: first 10 values = {py_fde[:10]}")

# Step 2: Index via OpenSearch ingest pipeline and read back
# Use existing pipeline (muvera-ingest-benchmark)
# Create a temp index
try:
    client.indices.delete(index=INDEX, ignore=[404])
except:
    pass

hnsw = {"name": "hnsw", "space_type": "innerproduct", "engine": "faiss",
        "parameters": {"ef_construction": 16, "m": 4}}
client.indices.create(index=INDEX, body={
    "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}},
    "mappings": {"properties": {
        "colbert_vectors": {"type": "object", "enabled": False},
        "muvera_fde": {"type": "knn_vector", "dimension": FDE_DIM, "method": hnsw}}}})

# Index the test doc through the pipeline
client.index(index=INDEX, id="test1", pipeline=PIPELINE,
             body={"colbert_vectors": test_vectors})
client.indices.refresh(index=INDEX)

# Read back the FDE
doc = client.get(index=INDEX, id="test1", _source_includes=["muvera_fde"])
java_fde = np.array(doc["_source"]["muvera_fde"])

print(f"Java FDE:   first 10 values = {java_fde[:10]}")
print(f"Python FDE: first 10 values = {py_fde[:10]}")

# Compare
diff = np.abs(py_fde - java_fde)
print(f"\nMax absolute diff: {diff.max():.10f}")
print(f"Mean absolute diff: {diff.mean():.10f}")
print(f"Cosine similarity: {np.dot(py_fde, java_fde) / (np.linalg.norm(py_fde) * np.linalg.norm(java_fde)):.10f}")

if diff.max() < 1e-4:
    print("\n✅ MATCH — Python and Java FDE encoders produce identical output")
else:
    print("\n❌ MISMATCH — encoders diverge")
    # Find first divergence point
    for i in range(len(py_fde)):
        if abs(py_fde[i] - java_fde[i]) > 1e-4:
            print(f"  First divergence at index {i}: py={py_fde[i]:.8f} java={java_fde[i]:.8f}")
            break

# Cleanup
client.indices.delete(index=INDEX, ignore=[404])
