from math import ceil, floor
import logging

logger = logging.getLogger(__name__)


def qps_to_replicas(
    target_qps, processing_time, max_qp_replica=1, target_utilization=0.7
):
    """Provide a rough estimate of the number of replicas to support a given
    load (queries per second)

    Args:
        target_qps (int): target queries per second that you want to support
        processing_time (float): the estimated amount of time (in seconds)
            your service call takes
        max_qp_replica (int): maximum number of concurrent queries per replica
        target_utilization (float): proportion of CPU utilization you think is ideal

    Returns:
        int: Number of estimated replicas required to support a target number of queries per second.
    """
    concurrent_queries = target_qps * processing_time / target_utilization
    replicas = ceil(concurrent_queries / max_qp_replica)
    logger.info(
        "Approximately {} replicas are estimated to support {} queries per second.".format(
            replicas, target_qps
        )
    )
    return replicas


def replicas_to_qps(
    num_replicas, processing_time, max_qp_replica=1, target_utilization=0.7
):
    """Provide a rough estimate of the queries per second supported by a number of replicas

    Args:
        num_replicas (int): number of replicas
        processing_time (float): the estimated amount of time (in seconds) your service call takes
        max_qp_replica (int): maximum number of concurrent queries per replica
        target_utilization (float): proportion of CPU utilization you think is ideal

    Returns:
        int: queries per second supported by the number of replicas
    """
    qps = floor(num_replicas * max_qp_replica * target_utilization / processing_time)
    logger.info(
        "Approximately {} queries per second are supported by {} replicas.".format(
            qps, num_replicas
        )
    )
    return qps


def nodes_to_replicas(n_cores_per_node, n_nodes=3, cpu_cores_per_replica=0.1):
    """Provide a rough estimate of the number of replicas supported by a
    given number of nodes with n_cores_per_node cores each

    Args:
        n_cores_per_node (int): Total number of cores per node within an AKS
            cluster that you want to use
        n_nodes (int): Number of nodes (i.e. VMs) used in the AKS cluster
        cpu_cores_per_replica (float): Cores assigned to each replica. This
            can be fractional and corresponds to the
            cpu_cores argument passed to AksWebservice.deploy_configuration()

    Returns:
        int: Total number of replicas supported by the configuration
    """
    n_cores_avail = (n_cores_per_node - 0.5) * n_nodes - 4.45
    replicas = floor(n_cores_avail / cpu_cores_per_replica)
    logger.info(
        "Approximately {} replicas are supported by {} nodes with {} cores each.".format(
            replicas, n_nodes, n_cores_per_node
        )
    )
    return replicas
