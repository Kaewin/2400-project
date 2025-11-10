"""
Brute Force Algorithm for Hamiltonian Path Problem

This module contains brute force implementations for solving the Hamiltonian 
cycle problem by checking all possible permutations.
"""

import numpy as np
from itertools import permutations
from typing import Tuple, List, Optional


def brute_force(graph) -> float:
    """
    Brute force algorithm for finding the minimum cost Hamiltonian cycle.
    Checks all possible permutations of nodes.
    
    Args:
        graph: Adjacency matrix where graph[i][j] is the weight from node i to j.
               Can be a numpy array or list of lists.
    
    Returns:
        float: The minimum cost of a Hamiltonian cycle, or inf if none exists.
    
    Time Complexity: O(n!)
    Space Complexity: O(n)
    """
    n = len(graph)
    
    # Start from node 0, permute the remaining nodes
    remaining_nodes = list(range(1, n))
    min_cost = float('inf')
    
    # Check all permutations
    for perm in permutations(remaining_nodes):
        # Build the path: 0 -> perm[0] -> perm[1] -> ... -> perm[n-2] -> 0
        path = [0] + list(perm)
        
        # Calculate cost of this path
        cost = 0
        valid = True
        
        for i in range(len(path)):
            current = path[i]
            next_node = path[(i + 1) % len(path)]  # Wrap around to complete cycle
            
            if graph[current][next_node] == 0:
                # No edge exists
                valid = False
                break
            
            cost += graph[current][next_node]
        
        if valid and cost < min_cost:
            min_cost = cost
    
    return min_cost


def brute_force_with_path(graph) -> Tuple[float, Optional[List[int]]]:
    """
    Brute force algorithm that returns both the cost and the actual path.
    
    Args:
        graph: Adjacency matrix where graph[i][j] is the weight from node i to j.
    
    Returns:
        Tuple[float, Optional[List[int]]]: 
            - Minimum cost of Hamiltonian cycle
            - List representing the path (or None if no cycle exists)
    """
    n = len(graph)
    
    remaining_nodes = list(range(1, n))
    min_cost = float('inf')
    best_path = None
    
    # Check all permutations
    for perm in permutations(remaining_nodes):
        # Build the path: 0 -> perm[0] -> perm[1] -> ... -> perm[n-2] -> 0
        path = [0] + list(perm) + [0]  # Add return to start
        
        # Calculate cost of this path
        cost = 0
        valid = True
        
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            if graph[current][next_node] == 0:
                valid = False
                break
            
            cost += graph[current][next_node]
        
        if valid and cost < min_cost:
            min_cost = cost
            best_path = path
    
    if min_cost == float('inf'):
        return min_cost, None
    
    return min_cost, best_path


def brute_force_optimized(graph) -> float:
    """
    Slightly optimized brute force using numpy and early termination.
    
    Args:
        graph: Adjacency matrix (numpy array preferred)
    
    Returns:
        float: Minimum cost of Hamiltonian cycle
    """
    if not isinstance(graph, np.ndarray):
        graph = np.array(graph)
    
    n = len(graph)
    remaining_nodes = list(range(1, n))
    min_cost = float('inf')
    
    # Check all permutations
    for perm in permutations(remaining_nodes):
        path = [0] + list(perm)
        
        # Calculate cost with early termination
        cost = 0
        valid = True
        
        for i in range(n):
            current = path[i]
            next_node = path[(i + 1) % n]
            
            if graph[current][next_node] == 0:
                valid = False
                break
            
            cost += graph[current][next_node]
            
            # Early termination if cost already exceeds minimum
            if cost >= min_cost:
                valid = False
                break
        
        if valid and cost < min_cost:
            min_cost = cost
    
    return min_cost


def brute_force_count_paths(graph) -> Tuple[float, int]:
    """
    Brute force that also counts total valid Hamiltonian cycles.
    
    Args:
        graph: Adjacency matrix
    
    Returns:
        Tuple[float, int]: Minimum cost and count of valid cycles
    """
    n = len(graph)
    remaining_nodes = list(range(1, n))
    min_cost = float('inf')
    valid_cycle_count = 0
    
    for perm in permutations(remaining_nodes):
        path = [0] + list(perm)
        
        cost = 0
        valid = True
        
        for i in range(n):
            current = path[i]
            next_node = path[(i + 1) % n]
            
            if graph[current][next_node] == 0:
                valid = False
                break
            
            cost += graph[current][next_node]
        
        if valid:
            valid_cycle_count += 1
            if cost < min_cost:
                min_cost = cost
    
    return min_cost, valid_cycle_count


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
        if len(graph.shape) != 2:
            return False
        
        n, m = graph.shape
        if n != m:
            return False
        
        if not np.allclose(np.diag(graph), 0):
            return False
        
        return True
    except:
        return False