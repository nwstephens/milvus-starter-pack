from typing import List
import numpy as np
from multiprocessing import Process
import pymilvus
import time
from minio import Minio
import os

from pymilvus import (
    connections, utility
)
from pymilvus.bulk_writer import LocalBulkWriter, BulkFileType  # pip install pymilvus[bulk_writer]

from milvus_common import (
    get_milvus_client, search_top_k, ID_FIELD, EMBEDDING_FIELD, compute_groundtruth, compute_recall,
    generate_vectors, setup_index, setup_milvus
)

from multiprocessing import Process

# minio
MINIO_PORT = 9001
MINIO_URL = f"localhost:{MINIO_PORT}"
MINIO_SECRET_KEY = "minioadmin"
MINIO_ACCESS_KEY = "minioadmin"

def upload_to_minio(file_path: List[List[str]], bucket_name="milvus-bucket"):
    minio_client = Minio(endpoint=MINIO_URL, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    for batch in file_path:
        for file in batch:
            minio_client.fput_object(bucket_name, file, file, num_parallel_uploads=5)
     
    
def ingest_data_bulk(collection_name, vectors, schema: pymilvus.CollectionSchema, log_times=True, use_local_bulk_writer=True, debug=False):
    print(f"-  Ingesting {len(vectors) // 1000}k vectors, Bulk")
    tic = time.perf_counter()
    collection = pymilvus.Collection(collection_name, using=get_milvus_client()._using)

    if use_local_bulk_writer:
        # # Prepare source data for faster ingestion
        writer = LocalBulkWriter(
            schema=schema,
            local_path='bulk_data',
            segment_size=512 * 1024 * 1024, # Default value
            file_type=BulkFileType.NPY
        )
        for id, vec in enumerate(vectors):
            writer.append_row({ID_FIELD: id, EMBEDDING_FIELD: vec})

        if debug:
            print(writer.batch_files)
        def callback(file_list):
            if debug:
                print(f"  -  Commit successful")
                print(file_list)
        writer.commit(call_back=callback)
        files_to_upload = writer.batch_files
    else:
        # Directly save NPY files
        np.save("bulk_data/embedding.npy", vectors)
        np.save("bulk_data/id.npy", np.arange(len(vectors)))
        files_to_upload = [["bulk_data/embedding.npy", "bulk_data/id.npy"]]
    
    toc = time.perf_counter()
    if log_times:
        print(f"  -  File save time: {toc - tic:.2f} seconds")
    # Import data
    upload_to_minio(files_to_upload)
    
    job_ids = [utility.do_bulk_insert(collection_name, batch, using=get_milvus_client()._using) for batch in files_to_upload]

    while True:
        tasks = [utility.get_bulk_insert_state(job_id, using=get_milvus_client()._using) for job_id in job_ids]
        success = all(task.state_name == "Completed" for task in tasks)
        failure = any(task.state_name == "Failed" for task in tasks)
        for i in range(len(tasks)):
            task = tasks[i]
            if debug:
                print(f"  -  Task {i}/{len(tasks)} state: {task.state_name}, Progress percent: {task.infos['progress_percent']}, Imported row count: {task.row_count}")
            if task.state_name == "Failed":
                print(task)
        if success or failure:
            break
        time.sleep(2)

    added_entities = str(sum([task.row_count for task in tasks]))
    failure = failure or added_entities != str(len(vectors))
    if failure:
        print(f"  -  Ingestion failed. Added entities: {added_entities}")
    toc = time.perf_counter()
    if log_times:
        print(f"-  Ingestion time: {toc - tic:.2f} seconds. Success: {success}, Failure: {failure}")

if __name__ == '__main__':
    collection_name = "milvus_tiny_example"
    dim = 512
    n_docs = int(1000e3)
    n_queries = 10000
    topk = 30

    schema = setup_milvus(collection_name, dim)
    vectors = generate_vectors(n_docs, dim, distribution="blobs")

    ingest_data_bulk(collection_name, vectors, schema)
    
    queries = generate_vectors(n_queries, dim, distribution="blobs")
    gt_indices = compute_groundtruth(vectors, queries, topk)

    for index_type in ["HNSW", "GPU_CAGRA", "GPU_IVF_PQ"]:
        print(index_type)
        setup_index(collection_name, index_type, n_docs, dim)
        search_result = search_top_k(collection_name, queries, topk, index_type, single_query=False)
        compute_recall(gt_indices, search_result, topk)
