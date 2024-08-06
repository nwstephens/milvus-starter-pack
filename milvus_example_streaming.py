from multiprocessing import Process
import pymilvus
import time
from pymilvus import (
    connections,
)
from milvus_common import (
    get_milvus_client, search_top_k, ID_FIELD, EMBEDDING_FIELD, compute_groundtruth, compute_recall,
    generate_vectors, setup_index, setup_milvus
)
from multiprocessing import Process


def ingest_data_bulk(collection_name, vectors, log_times=True):
    print(f"Ingesting {len(vectors) // 1000}k vectors, Bulk")
    tic = time.perf_counter()
    collection = pymilvus.Collection(collection_name, using=get_milvus_client()._using)
    entities = [
        {ID_FIELD: id, EMBEDDING_FIELD: vec} for id, vec in enumerate(vectors)
    ]
    collection.insert(entities)
    collection.flush()
    toc = time.perf_counter()
    if log_times:
        print(f"-  Ingestion time: {toc - tic:.2f} seconds")

def ingest_data_single_thread(collection_name, vectors, log_times=True):
    print(f"-  Ingesting {len(vectors) // 1000}k vectors, Single thread")
    tic = time.perf_counter()
    collection = pymilvus.Collection(collection_name, using=get_milvus_client()._using)
    batch_size = 10000
    for i in range(0, len(vectors), batch_size):
        entities = [
            {ID_FIELD: (id + i), EMBEDDING_FIELD: vec} for id, vec in enumerate(vectors[i:i+batch_size])
        ]
        collection.insert(entities)
    collection.flush()
    toc = time.perf_counter()
    if log_times:
        print(f"-  Ingestion time: {toc - tic:.2f} seconds")

def ingest_data_multi_thread(collection_name, vectors, log_times=True):
    print(f"-  Ingesting {len(vectors) // 1000}k vectors, Multi thread")
    batch_size = 10000
    pool_size = 5
    vecs_per_process = len(vectors) // pool_size
    def insert_vecs(id):
        collection = pymilvus.Collection(collection_name, using=get_milvus_client()._using)
        current_vecs_per_process = 0
        while current_vecs_per_process < vecs_per_process:
            start = id * vecs_per_process + current_vecs_per_process
            end = start + min(batch_size, vecs_per_process - current_vecs_per_process)
            # print("inserting items %s - %s" % (start, end))

            entities = [
                {ID_FIELD: (id + start), EMBEDDING_FIELD: vec} for id, vec in enumerate(vectors[start:end])
            ]
            collection.insert(entities)
            current_vecs_per_process = current_vecs_per_process + batch_size
        connections.disconnect("default")

    tic = time.perf_counter()
    processes = []
    for i in range(pool_size):
        p = Process(target=insert_vecs, args=(i,))
        p.start()
        processes.append(p)
    
    for i in processes:
        i.join()

    collection = pymilvus.Collection(collection_name, using=get_milvus_client()._using)
    collection.flush()
    toc = time.perf_counter()
    if log_times:
        print(f"-  Ingestion time: {toc - tic:.2f} seconds")


if __name__ == '__main__':
    collection_name = "milvus_tiny_example"
    dim = 512
    n_docs = int(1000e3)
    n_queries = 10000
    topk = 30

    setup_milvus(collection_name, dim)
    vectors = generate_vectors(n_docs, dim, distribution="blobs")
    # ingest_data_single_thread(collection_name, vectors)
    ingest_data_multi_thread(collection_name, vectors)
    queries = generate_vectors(n_queries, dim, distribution="blobs")
    gt_indices = compute_groundtruth(vectors, queries, topk)

    for index_type in ["HNSW", "GPU_CAGRA", "GPU_IVF_PQ"]:
        print(index_type)
        setup_index(collection_name, index_type, n_docs, dim)
        search_result = search_top_k(collection_name, queries, topk, index_type, single_query=False)
        compute_recall(gt_indices, search_result, topk)
