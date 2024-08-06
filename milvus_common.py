import numpy as np
import cupy as cp
from pylibraft.neighbors import brute_force
from sklearn.datasets import make_blobs
import pymilvus
import time
from pymilvus import (
    FieldSchema, CollectionSchema, DataType
)

MILVUS_PORT = 19531
MILVUS_HOST = f"http://localhost:{MILVUS_PORT}"
ID_FIELD="id"
EMBEDDING_FIELD="embedding"

def get_milvus_client():
    return pymilvus.MilvusClient(uri=MILVUS_HOST)

def generate_vectors(n_docs: int, dim: int, distribution="blobs"):
    if distribution == "uniform":
        rng = np.random.default_rng()
        return np.random.rand(n_docs, dim)
    elif distribution == "blobs":
        X, _ = make_blobs(n_samples=n_docs, centers=100, n_features=dim, random_state=42)
        return X

def search_top_k(collection_name, queries, top_k, index_type, log_times=True, single_query=False):
    tic = time.perf_counter()
    collection = pymilvus.Collection(collection_name, using=get_milvus_client()._using)
    QUERY_PARAMS = {
        "IVF_PQ": dict(nprobe=20),
        "GPU_IVF_PQ": dict(nprobe=30),
        "IVF_FLAT": dict(nprobe=16),
        "GPU_IVF_FLAT": dict(nprobe=16),
        "HNSW": dict(ef=256),
        "GPU_CAGRA": dict(itopk=96),
        "NONE": dict(),
    }
    if single_query:
        result = []
        for query in queries:
            result.append(collection.search(
                data=[query], anns_field=EMBEDDING_FIELD, param=QUERY_PARAMS[index_type], limit=top_k
            ))
    else:
        result = collection.search(
            data=queries, anns_field=EMBEDDING_FIELD, param=QUERY_PARAMS[index_type], limit=top_k
        )
    toc = time.perf_counter()
    if log_times:
        params = QUERY_PARAMS[index_type]
        print(f"-  Search time: {toc - tic:.2f} seconds . ({params} - Single query: {single_query})")
    return result

def compute_groundtruth(vectors, queries, k):
    """
    brute_force_index = brute_force.build(index_device, "euclidean")
    brute_force.search(
        brute_force_index,
        queries_device,
        k,
        neighbors=indices_device,
        distances=distances_device,
    )
    """
    _, neighbors = brute_force.knn(cp.array(vectors, dtype=cp.float32), cp.array(queries, dtype=cp.float32), k, metric="euclidean")
    #brute_force.knn(index_device, queries_device, k, indices_device, distances_device, metric="euclidean")
    return neighbors.copy_to_host()

def compute_recall(gt_indices, search_res, topk):
    result = np.zeros(shape=(len(search_res), topk), dtype=np.int64)
    for i, res in enumerate(search_res):
        # Handle single_query
        if type(res) == pymilvus.client.abstract.SearchResult:
            res = res[0]
        if len(res.ids) < topk:
            res.ids += [-1] * (topk - len(res.ids))
        result[i, :] = res.ids

    n = 0
    for i in range(result.shape[0]):
        n += np.intersect1d(result[i, :], gt_indices[i, :]).size
    recall = n / result.size
    print(f"-  Recall: {recall:.3f}")
    return recall

def setup_index(collection_name, index_type: str, n_docs: int, dim: int, drop=True, log_times=True):
    collection = pymilvus.Collection(collection_name, using=get_milvus_client()._using)
    INDEX_PARAMS = {
        "IVF_PQ": dict(
            index_type="IVF_PQ",
            metric_type="L2",
            params={"nlist": min(int(np.sqrt(n_docs)), 5000), "m": dim // 4},
        ),
        "GPU_IVF_PQ": dict(
            index_type="GPU_IVF_PQ",
            metric_type="L2",
            params={"nlist": min(int(np.sqrt(n_docs)), 5000), "m": dim // 4},
        ),
        "IVF_FLAT": dict(
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": min(int(np.sqrt(n_docs)), 5000)},
        ),
        "GPU_IVF_FLAT": dict(
            index_type="GPU_IVF_FLAT",
            metric_type="L2",
            params={"nlist": min(int(np.sqrt(n_docs)), 5000)},
        ),
        "HNSW": dict(
            index_type="HNSW",
            metric_type="L2",
            params={"M": 16, "efConstruction": 600},
        ),
        "GPU_CAGRA": dict(
            index_type="GPU_CAGRA",
            metric_type="L2",
            params={"graph_degree": 32, "intermediate_graph_degree": 96, "build_algo": "NN_DESCENT"},
        ),
    }
    if collection.has_index():
        if drop:
            print("Dropping existing index.")
            collection.release()
            collection.drop_index()
        else:
            raise ValueError("Index already exists.")
        
    tic = time.perf_counter()
    collection.create_index(field_name=EMBEDDING_FIELD, index_params=INDEX_PARAMS[index_type])
    collection.load()
    toc = time.perf_counter()
    if log_times:
        params = INDEX_PARAMS[index_type]["params"]
        print(f"-  Index creation time: {toc - tic:.2f} seconds. ({params})")

def setup_milvus(collection_name: str, dim=128, drop=True):
    client = get_milvus_client()
    fields = [
        FieldSchema(name=ID_FIELD, dtype=DataType.INT64, is_primary=True, dim=dim),
        FieldSchema(name=EMBEDDING_FIELD, dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]

    schema = CollectionSchema(fields)
    schema.verify()

    if collection_name in client.list_collections():
        if drop:
            print(f"Collection '{collection_name}' already exists. Deleting collection...")
            client.drop_collection(collection_name)
        else:
            raise ValueError(f"Collection '{collection_name}' already exists.")

    client.create_collection(collection_name, schema=schema)
    return schema
