"""
consensus_pipeline.py
======================

Consensus clustering routine for connectome subgraphs.
Implements the iterative consensus approach of Lancichinetti & Fortunato (2012)
with dissensus analysis based on Peixoto (2021).

Algorithm overview
------------------
1. Run a stochastic community-detection algorithm r times on G.
2. Build the consensus matrix D_ij = fraction of runs where nodes i,j
   co-occurred in the same community.
3. Threshold D at tau, forming a weighted consensus graph G'.
4. Check convergence: stop if G' is a disjoint union of cliques.
   Otherwise repeat from step 1 using G' as the new graph.
5. At convergence, each connected component of G' defines one community.

Dissensus analysis (Peixoto 2021)
----------------------------------
When the ensemble of partitions is multimodal, a single consensus can be
misleading. `dissensus_analysis` clusters the partitions themselves via
hierarchical clustering on pairwise ARI distances, revealing competing modes.

Usage
-----
    from consensus_pipeline import run_consensus_clustering
    import networkx as nx
    import community as community_louvain

    G = nx.karate_club_graph()
    algo = lambda G: community_louvain.best_partition(G)
    consensus_partition, history = run_consensus_clustering(G, algo)

References
----------
- Lancichinetti, A. & Fortunato, S. (2012). Consensus clustering in complex
  networks. Sci. Rep. 2, 336.
- Fortunato, S. & Hric, D. (2016). Community detection in networks: A user
  guide. Phys. Rep. 659, 1–44. (Section 4.2)
- Peixoto, T. P. (2021). Revealing consensus and dissensus between network
  partitions. Phys. Rev. X 11, 021003.

Author: Gutierrez Lab, Barnard College
"""

from __future__ import annotations

import warnings
from collections import Counter
from itertools import combinations
from typing import Callable, Optional

import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# ══════════════════════════════════════════════════════════════════════════════
# CORE ALGORITHM COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def build_consensus_matrix(
    partitions: list[dict],
    nodes: list,
) -> np.ndarray:
    """
    Build the consensus (co-occurrence) matrix D.

    D_ij = fraction of partitions in which nodes i and j are in the same
    community. D is symmetric, values in [0, 1], and D_ii = 1 for all i.

    Parameters
    ----------
    partitions : list of dict {node: community_id}
        Output of r independent clustering runs.
    nodes : list
        Ordered list of node identifiers defining row/column order.

    Returns
    -------
    D : np.ndarray, shape (N, N)

    Raises
    ------
    ValueError
        If partitions is empty or any partition is missing nodes.
    """
    if not partitions:
        raise ValueError("partitions must be non-empty.")

    N = len(nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}

    # Validate: warn if any partition is missing nodes
    for k, part in enumerate(partitions):
        missing = set(nodes) - set(part.keys())
        if missing:
            warnings.warn(
                f"Partition {k} is missing {len(missing)} node(s). "
                "Missing nodes will be ignored.",
                UserWarning,
            )

    D = np.zeros((N, N), dtype=np.float64)

    for part in partitions:
        comm_to_nodes: dict[int, list[int]] = {}
        for node, comm in part.items():
            idx = node_idx.get(node)
            if idx is not None:
                comm_to_nodes.setdefault(comm, []).append(idx)

        for idxs in comm_to_nodes.values():
            for a in idxs:
                for b in idxs:
                    D[a, b] += 1.0

    D /= len(partitions)
    return D


def threshold_consensus_matrix(
    D: np.ndarray,
    nodes: list,
    tau: float,
) -> tuple[nx.Graph, int]:
    """
    Build a weighted graph from the consensus matrix by thresholding.

    Edges with D_ij < tau are removed. Surviving edges have weight = D_ij.

    Parameters
    ----------
    D     : np.ndarray (N, N) consensus matrix
    nodes : ordered list of N node identifiers
    tau   : float in [0, 1]; edges with D_ij < tau are removed

    Returns
    -------
    G_consensus : nx.Graph
    n_isolated  : int, number of zero-degree nodes after thresholding
    """
    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"tau must be in [0, 1], got {tau}.")

    G_consensus = nx.Graph()
    G_consensus.add_nodes_from(nodes)
    N = len(nodes)

    for i in range(N):
        for j in range(i + 1, N):
            if D[i, j] >= tau:
                G_consensus.add_edge(nodes[i], nodes[j], weight=D[i, j])

    n_isolated = sum(1 for n in G_consensus.nodes()
                     if G_consensus.degree(n) == 0)
    return G_consensus, n_isolated


def is_clique_union(G: nx.Graph) -> bool:
    """
    Return True if G is a disjoint union of cliques (convergence criterion).

    A connected component C is a clique iff |E(C)| == |C| * (|C|-1) / 2.
    Single-node components trivially satisfy this.
    """
    for comp in nx.connected_components(G):
        n = len(comp)
        subg = G.subgraph(comp)
        if subg.number_of_edges() != n * (n - 1) // 2:
            return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_consensus_clustering(
    G_init: nx.Graph,
    base_algorithm: Callable[[nx.Graph], dict],
    tau: float = 0.5,
    r: int = 20,
    max_iter: int = 10,
    verbose: bool = True,
) -> tuple[dict, list[dict]]:
    """
    Iterative consensus clustering (Lancichinetti & Fortunato, 2012).

    Parameters
    ----------
    G_init : nx.Graph
        The graph to cluster. May be weighted or unweighted.
    base_algorithm : callable (nx.Graph) -> dict {node: community_id}
        Stochastic community-detection function. Must accept a weighted
        nx.Graph (edge attribute 'weight') and return a partition dict.
    tau : float, default 0.5
        Co-occurrence threshold. Pairs of nodes that appear in the same
        community in fewer than tau * r runs are disconnected in the
        consensus graph.
    r : int, default 20
        Number of runs per iteration.
    max_iter : int, default 10
        Maximum number of iterations before stopping regardless of convergence.
    verbose : bool, default True
        Print per-iteration diagnostics.

    Returns
    -------
    consensus_partition : dict {node: community_id}
        Final stable partition. Community IDs are arbitrary integers.
    history : list of dict
        Per-iteration diagnostics with keys:
        iteration, n_edges_D, n_isolated, n_components, converged.

    Notes
    -----
    If convergence is not reached within max_iter, the partition at the last
    iteration is returned. A warning is issued in this case.
    The base_algorithm is called with the consensus-weighted graph from the
    previous iteration (or the original G_init for the first iteration).
    """
    nodes = sorted(G_init.nodes())
    G_curr = G_init.copy()
    history: list[dict] = []
    converged_flag = False

    for iteration in range(1, max_iter + 1):
        runs = [base_algorithm(G_curr) for _ in range(r)]

        D = build_consensus_matrix(runs, nodes)

        G_cons, n_iso = threshold_consensus_matrix(D, nodes, tau)

        converged = is_clique_union(G_cons)
        n_comp = nx.number_connected_components(G_cons)

        diag = {
            "iteration":    iteration,
            "n_edges_D":    G_cons.number_of_edges(),
            "n_isolated":   n_iso,
            "n_components": n_comp,
            "converged":    converged,
        }
        history.append(diag)

        if verbose:
            print(
                f"  Iter {iteration:2d}: edges={G_cons.number_of_edges():5d}  "
                f"isolated={n_iso:3d}  components={n_comp:3d}  "
                f"converged={converged}"
            )

        if converged:
            converged_flag = True
            break

        G_curr = G_cons

    if not converged_flag:
        warnings.warn(
            f"Consensus clustering did not converge within {max_iter} iterations. "
            "Consider increasing max_iter or adjusting tau.",
            UserWarning,
        )

    # Extract partition from connected components of final consensus graph
    consensus_partition: dict = {}
    for comm_id, comp in enumerate(nx.connected_components(G_cons)):
        for node in comp:
            consensus_partition[node] = comm_id

    return consensus_partition, history


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def partition_to_array(
    partition: dict,
    nodes: list,
) -> np.ndarray:
    """Convert a partition dict to an array aligned to `nodes` order."""
    return np.array([partition[n] for n in nodes])


def pairwise_ari_list(arrays: list[np.ndarray]) -> list[float]:
    """Return a flat list of ARI values for all unique pairs of arrays."""
    return [adjusted_rand_score(a, b) for a, b in combinations(arrays, 2)]


def run_tau_sweep(
    G: nx.Graph,
    base_algorithm: Callable,
    tau_range: Optional[np.ndarray] = None,
    r: int = 20,
    max_iter: int = 10,
    ground_truth: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Sweep tau values and return a DataFrame of results.

    Parameters
    ----------
    G              : nx.Graph
    base_algorithm : callable
    tau_range      : array-like of float; default np.arange(0.2, 0.85, 0.1)
    r              : runs per iteration
    max_iter       : max iterations
    ground_truth   : optional dict {node: community_id}; if provided, ARI and
                     NMI vs. ground truth are included in output

    Returns
    -------
    pd.DataFrame with columns: tau, k, iters, [ARI, NMI if ground_truth given]
    """
    if tau_range is None:
        tau_range = np.round(np.arange(0.2, 0.85, 0.1), 2)

    nodes = sorted(G.nodes())
    gt_array = (partition_to_array(ground_truth, nodes)
                if ground_truth is not None else None)

    records = []
    for tau in tau_range:
        part, hist = run_consensus_clustering(
            G, base_algorithm, tau=float(tau), r=r,
            max_iter=max_iter, verbose=False,
        )
        rec = {
            "tau":   tau,
            "k":     len(set(part.values())),
            "iters": len(hist),
        }
        if gt_array is not None:
            arr = partition_to_array(part, nodes)
            rec["ARI"] = adjusted_rand_score(gt_array, arr)
            rec["NMI"] = normalized_mutual_info_score(
                gt_array, arr, average_method="arithmetic"
            )
        records.append(rec)
    return pd.DataFrame(records)


def partition_stability(
    G: nx.Graph,
    base_algorithm: Callable,
    n_reps: int = 10,
    tau: float = 0.5,
    r: int = 20,
    max_iter: int = 10,
) -> tuple[list[float], list[np.ndarray]]:
    """
    Run the full consensus procedure n_reps times and measure pairwise ARI.

    Returns
    -------
    ari_pairs : list of float (n_reps*(n_reps-1)/2 values)
    arrays    : list of np.ndarray, one per rep (consensus partition as array)
    """
    nodes = sorted(G.nodes())
    arrays: list[np.ndarray] = []
    for _ in range(n_reps):
        part, _ = run_consensus_clustering(
            G, base_algorithm, tau=tau, r=r,
            max_iter=max_iter, verbose=False,
        )
        arrays.append(partition_to_array(part, nodes))
    ari_pairs = pairwise_ari_list(arrays)
    return ari_pairs, arrays


def dissensus_analysis(
    partitions: list[dict],
    nodes: list,
    cut_height: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Detect multiple partition modes via hierarchical clustering of ARI distances.

    Implements the partition-clustering approach described in Peixoto (2021):
    partitions are treated as data points, pairwise ARI as similarity, and
    Ward linkage is applied to the distance matrix (1 - ARI) to detect modes.

    Parameters
    ----------
    partitions  : list of dicts {node: community_id}
    nodes       : ordered list of node identifiers
    cut_height  : float, Ward linkage height at which to cut dendrogram

    Returns
    -------
    mode_labels : np.ndarray (n_partitions,), mode assignment per partition
    Z           : linkage matrix (for plotting with scipy.dendrogram)
    n_modes     : int, number of distinct modes detected
    """
    arrays = [partition_to_array(p, nodes) for p in partitions]
    n = len(arrays)

    ari_mat = np.array([
        [adjusted_rand_score(arrays[i], arrays[j]) for j in range(n)]
        for i in range(n)
    ])
    np.fill_diagonal(ari_mat, 1.0)

    dist_condensed = squareform(1 - ari_mat, checks=False)
    Z = linkage(dist_condensed, method="ward")
    mode_labels = fcluster(Z, t=cut_height, criterion="distance")
    return mode_labels, Z, len(set(mode_labels))


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_partition(
    partition: dict,
    nodes: list,
    label: str = "Partition",
) -> None:
    """
    Run sanity checks on a partition dict.

    Checks
    ------
    - All expected nodes are present
    - No NaN community labels
    - Community IDs are integers
    - At least 2 communities (non-trivial partition)
    - Warns if any community contains a single node

    Raises
    ------
    ValueError on fatal errors.
    """
    missing = set(nodes) - set(partition.keys())
    if missing:
        raise ValueError(
            f"{label}: {len(missing)} node(s) missing from partition."
        )

    labels = list(partition.values())

    if any(l is None or (isinstance(l, float) and np.isnan(l)) for l in labels):
        raise ValueError(f"{label}: partition contains NaN/None community labels.")

    if not all(isinstance(l, (int, np.integer)) for l in labels):
        warnings.warn(
            f"{label}: community labels are not all integers. "
            "Conversion may cause unexpected behaviour.",
            UserWarning,
        )

    k = len(set(labels))
    if k < 2:
        raise ValueError(
            f"{label}: trivial partition — all nodes in one community."
        )

    sizes = Counter(labels)
    singletons = [c for c, s in sizes.items() if s == 1]
    if singletons:
        warnings.warn(
            f"{label}: {len(singletons)} singleton community(ies) detected.",
            UserWarning,
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT (for quick testing)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import community.community_louvain as community_louvain

    print("=== consensus_pipeline.py — quick test on Karate Club graph ===\n")

    G = nx.karate_club_graph()
    algo = lambda g: community_louvain.best_partition(g, random_state=None)

    print("Running consensus clustering (tau=0.5, r=20)...")
    part, hist = run_consensus_clustering(G, algo, tau=0.5, r=20, verbose=True)

    k = len(set(part.values()))
    print(f"\nConsensus partition: {k} communities")
    print(f"Iterations: {len(hist)}")

    print("\nRunning tau sweep (0.2 → 0.8)...")
    df = run_tau_sweep(G, algo, r=10)
    print(df.to_string(index=False))

    print("\nDissensus analysis (20 single-run partitions)...")
    nodes = sorted(G.nodes())
    runs = [algo(G) for _ in range(20)]
    mode_labels, Z, n_modes = dissensus_analysis(runs, nodes, cut_height=0.3)
    print(f"Modes detected: {n_modes}")

    print("\nDone.")
