# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Dict, Tuple

import jax.numpy as jnp

from pyhgf.typing import DirichletNode, Edges


def dirichlet_input_prediction_error(
    edges: Edges,
    attributes: Dict,
    value: float,
    node_idx: int,
    dirichlet_node: DirichletNode,
    input_nodes_idx,
    **args,
) -> Tuple:
    """Prediction error and update the child networks of a Dirichlet process node.

    When receiving a new input, this node chose to either:
    1. Allocate the value to a pre-existing cluster.
    2. Create a new cluster.

    The network always contains a temporary branch as the new cluster
    candidate. This branch is parametrized under the new observation to assess its
    likelihood and the previous clusters' likelihood.

    Parameters
    ----------
    edges :
        The edges of the neural network as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index lists the value/volatility parents/children.
    attributes :
        The attributes of the probabilistic nodes.
    value :
        The new observed value(s). The input shape should match the input shape
        at the child level.
    node_idx :
        Pointer to the Dirichlet process input node.
    dirichlet_node :
        Static parameters of the Dirichlet process node.
    input_nodes_idx :
        Static input nodes' parameters for the neural network.

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the neural network.
    input_nodes_idx :
        Static input nodes' parameters for the neural network.
    dirichlet_node :
        Static parameters of the Dirichlet process node.

    """
    # unpack static parameters from the Dirichlet node
    (
        base_network,
        create_cluster_fn,
        log_likelihood_fn,
        parametrize_cluster_fn,
        cluster_input_idxs,
    ) = dirichlet_node

    value = value  # the input value
    alpha = attributes[node_idx]["alpha"]  # the concentration parameter

    # likelihood of the current observation under possible clusters
    # -------------------------------------------------------------

    # the temporary cluster is always the last created
    temp_idx = cluster_input_idxs[-1]

    # parametrize the temporary branch given the current observed value
    attributes = parametrize_cluster_fn(
        value=value, input_idx=temp_idx, attributes=attributes, edges=edges
    )

    # evaluate the likelihood of the current observation under all available branches
    # i.e. pre-existing cluster and temporary one
    clusters_log_likelihood = []
    for input_idx in cluster_input_idxs:
        likelihood = log_likelihood_fn(
            input_idx=input_idx, value=value, attributes=attributes, edges=edges
        )
        clusters_log_likelihood.append(likelihood)

    # probability of being assigned to a pre-existing cluster
    pi_clusters = [
        n / (alpha + attributes[node_idx]["n_total"]) for n in attributes[node_idx]["n"]
    ]

    # probability to draw a new cluster
    pi_new = alpha / (alpha + attributes[node_idx]["n_total"])

    # the probability for a new cluster is attributed to the temporary cluster
    pi_clusters[-1] = pi_new

    # the joint log-likelihoods (evidence + probability)
    clusters_log_likelihood = jnp.array(clusters_log_likelihood) + jnp.log(
        jnp.array(pi_clusters)
    )

    # decide which branch should be updated
    update_idx = jnp.argmax(clusters_log_likelihood)

    # belief propagation
    # ------------------

    # increment the number of observations for the given branch
    attributes[node_idx]["n"][update_idx] += 1

    # mark all branches unobserved
    for input_idx in cluster_input_idxs:
        attributes[input_idx]["observed"] = 0.0

    # if a new cluster was created, create a new temporary one
    if update_idx == attributes[node_idx]["n_clusters"]:
        attributes[node_idx]["n_clusters"] += 1
        attributes, edges, input_nodes_idx, dirichlet_node = create_cluster_fn(
            attributes=attributes,
            edges=edges,
            input_nodes_idx=input_nodes_idx,
            base_network=base_network,
            dirichlet_node_idx=node_idx,
            dirichlet_node=dirichlet_node,
        )
    else:
        # otherwise, pass the new observation and
        # ensure that the beliefs will propagate in the branch
        update_branch_idx = cluster_input_idxs[int(update_idx)]
        attributes[update_branch_idx]["observed"] = 1.0
        attributes[update_branch_idx]["value"] = value

    attributes[node_idx]["n_total"] += 1

    return attributes, edges, input_nodes_idx, dirichlet_node
