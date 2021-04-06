# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from reco_utils.common.k8s_utils import (
    qps_to_replicas,
    replicas_to_qps,
    nodes_to_replicas,
)


def test_qps_to_replicas():
    replicas = qps_to_replicas(target_qps=25, processing_time=0.1)
    assert replicas == 4


def test_replicas_to_qps():
    qps = replicas_to_qps(num_replicas=4, processing_time=0.1)
    assert qps == 27


def test_nodes_to_replicas():
    max_replicas = nodes_to_replicas(
        n_cores_per_node=4, n_nodes=3, cpu_cores_per_replica=0.1
    )
    assert max_replicas == 60
