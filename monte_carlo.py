"""
Monte Carlo Heuristic for Hamiltonian Cycle / TSP

This module provides a randomized (Monte Carlo) heuristic for approximating
the minimum-cost Hamiltonian cycle in a weighted graph.

It repeatedly samples random tours (permutations of nodes) starting at node 0
and keeps track of the best one seen so far.

This is NOT guaranteed to find the optimal solution, but it is usually much
faster than brute force and can scale to larger graphs.
"""

import numpy as np
from typing import Tuple, Optional, List


def monte_carlo(
    graph,
    iterations: int = 10_000,
    random_seed: Optional[int] = None
) -> float:
    """
    Monte Carlo heuristic for finding a (near-)minimum Hamiltonian cycle cost.

    Args:
        graph:
            Adjacency matrix where graph[i][j] is the weight from node i to j.
            Can be a numpy array or list of lists.
        iterations:
            Number of random tours to sample.
        random_seed:
            Optional seed for reproducibility.

    Returns:
        float: Best (minimum) tour cost found, or inf if none exists.
    """
    if not isinstance(graph, np.ndarray):
        graph = np.array(graph)

    n = len(graph)
    if n <= 1:
        return float("inf")

    if random_seed is not None:
        np.random.seed(random_seed)

    nodes = np.arange(1, n)  # we fix start at node 0
    best_cost = float("inf")

    for _ in range(iterations):
        # Random permutation of the remaining nodes
        perm = np.random.permutation(nodes)
        path = np.concatenate(([0], perm, [0]))

        cost = 0.0
        valid = True

        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            w = graph[u, v]
            if w <= 0:  # treat 0 / non-positive as "no edge"
                valid = False
                break
            cost += w

            # small early stopping optimization
            if cost >= best_cost:
                valid = False
                break

        if valid and cost < best_cost:
            best_cost = cost

    return best_cost


def monte_carlo_with_path(
    graph,
    iterations: int = 10_000,
    random_seed: Optional[int] = None
) -> Tuple[float, Optional[List[int]]]:
    """
    Monte Carlo heuristic that also returns the best path found.

    Args:
        graph:
            Adjacency matrix where graph[i][j] is the weight from node i to j.
        iterations:
            Number of random tours to sample.
        random_seed:
            Optional seed for reproducibility.

    Returns:
        (best_cost, best_path)

        best_cost:
            Minimum cost found (or inf if no valid tour exists).
        best_path:
            List of node indices describing the tour, including return to 0,
            e.g. [0, 3, 2, 1, 0]. None if no valid tour was found.
    """
    if not isinstance(graph, np.ndarray):
        graph = np.array(graph)

    n = len(graph)
    if n <= 1:
        return float("inf"), None

    if random_seed is not None:
        np.random.seed(random_seed)

    nodes = np.arange(1, n)
    best_cost = float("inf")
    best_path: Optional[List[int]] = None

    for _ in range(iterations):
        perm = np.random.permutation(nodes)
        path = np.concatenate(([0], perm, [0]))

        cost = 0.0
        valid = True

        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            w = graph[u, v]
            if w <= 0:
                valid = False
                break
            cost += w

            if cost >= best_cost:
                valid = False
                break

        if valid and cost < best_cost:
            best_cost = cost
            best_path = path.tolist()

    if best_cost == float("inf"):
        return best_cost, None

    return best_cost, best_path
