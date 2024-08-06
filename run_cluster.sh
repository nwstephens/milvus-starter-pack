#!/usr/bin/env bash

start() {
    echo "Starting Milvus Cluster..."
    microk8s helm install --wait milvus-cluster milvus/milvus -f chart.yml --set indexNode.replicas=1 --set queryNode.replicas=1
    microk8s kubectl port-forward service/milvus-cluster 19531:19530&
    microk8s kubectl port-forward service/milvus-cluster-minio 9001:9000&
}

stop() {
    microk8s helm uninstall milvus-cluster
    #kill %1
    #kill %2
}

case $1 in
    start)
        start
        ;;
    stop)
        stop
        ;;
    *)
        echo "please use bash milvus_cluster.sh start|stop"
        ;;
esac
