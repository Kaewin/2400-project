"""
Held-Karp Algorithm for Hamiltonian Path Problem

This module contains the complete implementation of the Held-Karp algorithm
for solving the Hamiltonian cycle problem using dynamic programming.
"""

import numpy as np
from typing import Tuple, List, Optional


def held_karp(graph) -> float:
    """
    Held-Karp algorithm for finding the minimum cost Hamiltonian cycle.
    
    Args:
        graph: Adjacency matrix where graph[i][j] is the weight from node i to j.
               Can be a numpy array or list of lists.
    
    Returns:
        float: The minimum cost of a Hamiltonian cycle, or inf if none exists.
    
    Time Complexity: O(n^2 * 2^n)
    Space Complexity: O(n * 2^n)
    """
    n = len(graph)
    
    # Initialize DP table: dp[mask][u] = min cost to reach node u with visited set mask
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Start at node 0
    
    # Fill DP table
    for mask in range(1 << n):
        for u in range(n):
            if mask & (1 << u):  # If node u is in the current set
                for v in range(n):
                    if mask & (1 << v) and graph[v][u]:  # If node v is in set and edge exists
                        dp[mask][u] = min(dp[mask][u], dp[mask ^ (1 << u)][v] + graph[v][u])
    
    # Find minimum cost to complete the cycle
    full_mask = (1 << n) - 1
    min_cost = min(dp[full_mask][u] + graph[u][0] for u in range(n) if graph[u][0])
    
    return min_cost


def held_karp_with_path(graph) -> Tuple[float, Optional[List[int]]]:
    """
    Held-Karp algorithm that returns both the cost and the actual path.
    
    Args:
        graph: Adjacency matrix where graph[i][j] is the weight from node i to j.
    
    Returns:
        Tuple[float, Optional[List[int]]]: 
            - Minimum cost of Hamiltonian cycle
            - List representing the path (or None if no cycle exists)
    """
    n = len(graph)
    
    # Initialize DP table and parent tracking
    dp = [[float('inf')] * n for _ in range(1 << n)]
    parent = [[None] * n for _ in range(1 << n)]
    dp[1][0] = 0
    
    # Fill DP table with path tracking
    for mask in range(1 << n):
        for u in range(n):
            if mask & (1 << u):
                for v in range(n):
                    if mask & (1 << v) and graph[v][u]:
                        new_cost = dp[mask ^ (1 << u)][v] + graph[v][u]
                        if new_cost < dp[mask][u]:
                            dp[mask][u] = new_cost
                            parent[mask][u] = v
    
    # Find the ending node with minimum cost
    full_mask = (1 << n) - 1
    min_cost = float('inf')
    last_node = None
    
    for u in range(n):
        if graph[u][0]:
            cost = dp[full_mask][u] + graph[u][0]
            if cost < min_cost:
                min_cost = cost
                last_node = u
    
    if min_cost == float('inf'):
        return min_cost, None
    
    # Reconstruct path
    path = []
    mask = full_mask
    current = last_node
    
    while current is not None:
        path.append(current)
        prev = parent[mask][current]
        if prev is not None:
            mask ^= (1 << current)
        current = prev
    
    path.reverse()
    path.append(0)  # Return to start to complete cycle
    
    return min_cost, path


def held_karp_optimized(graph) -> float:
    """
    Optimized version using numpy for better performance.
    
    Args:
        graph: Adjacency matrix (numpy array preferred)
    
    Returns:
        float: Minimum cost of Hamiltonian cycle
    """
    # Convert to numpy array if not already
    if not isinstance(graph, np.ndarray):
        graph = np.array(graph)
    
    n = len(graph)
    
    # Initialize DP table
    dp = np.full((1 << n, n), float('inf'), dtype=np.float64)
    dp[1][0] = 0
    
    # Dynamic programming
    for mask in range(1 << n):
        for u in range(n):
            if mask & (1 << u):
                for v in range(n):
                    if mask & (1 << v) and graph[v][u] > 0:
                        new_mask = mask ^ (1 << u)
                        if dp[new_mask][v] + graph[v][u] < dp[mask][u]:
                            dp[mask][u] = dp[new_mask][v] + graph[v][u]
    
    # Find minimum cost to complete the cycle
    full_mask = (1 << n) - 1
    min_cost = float('inf')
    
    for u in range(n):
        if graph[u][0] > 0:
            cost = dp[full_mask][u] + graph[u][0]
            if cost < min_cost:
                min_cost = cost
    
    return min_cost


def load_graph(filename: str) -> np.ndarray:
    """
    Load a graph from a numpy file.
    
    Args:
        filename: Path to .npy or .npz file
    
    Returns:
        np.ndarray: Adjacency matrix
    """
    if filename.endswith('.npz'):
        data = np.load(filename)
        # Assume the first array in the file is the graph
        graph = data[data.files[0]]
    else:
        graph = np.load(filename)
    
    return graph


def validate_graph(graph) -> bool:
    """
    Validate that the graph is a valid adjacency matrix.
    
    Args:
        graph: Graph to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Check if it's a square matrix
        if len(graph.shape) != 2:
            return False
        
        n, m = graph.shape
        if n != m:
            return False
        
        # Check diagonal is zero (no self-loops)
        if not np.allclose(np.diag(graph), 0):
            return False
        
        return True
    except:
        return False
