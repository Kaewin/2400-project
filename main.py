"""
Main benchmark script for Held-Karp algorithm.

This script imports pre-designed graphs, benchmarks the Held-Karp algorithm,
and outputs the results to a CSV file.
"""

import numpy as np
import time
import csv
import os
import sys
from pathlib import Path
from held_karp import held_karp, held_karp_with_path, held_karp_optimized, load_graph, validate_graph


def benchmark_graph(graph, graph_name="unknown", use_optimized=True):
    """
    Benchmark the Held-Karp algorithm on a single graph.
    
    Args:
        graph: Adjacency matrix (numpy array)
        graph_name: Name identifier for the graph
        use_optimized: Whether to use the optimized version
    
    Returns:
        dict: Benchmark results containing:
            - graph_name: Name of the graph
            - size: Number of nodes
            - states: Number of DP states (2^n)
            - memory_mb: Estimated memory usage in MB
            - time_seconds: Execution time in seconds
            - min_cost: Minimum Hamiltonian cycle cost
            - path_length: Length of the path (if found)
            - success: Whether the algorithm completed successfully
            - error: Error message if failed
    """
    n = len(graph)
    
    result = {
        'graph_name': graph_name,
        'size': n,
        'states': 2**n,
        'memory_mb': (2**n * n * 8) / (1024**2),
        'time_seconds': None,
        'min_cost': None,
        'path_length': None,
        'success': False,
        'error': None
    }
    
    try:
        # Choose algorithm version
        algorithm = held_karp_optimized if use_optimized else held_karp
        
        # Benchmark execution time
        start_time = time.perf_counter()
        min_cost = algorithm(graph)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        # Record results
        result['time_seconds'] = execution_time
        result['min_cost'] = min_cost if min_cost != float('inf') else None
        result['path_length'] = n + 1 if min_cost != float('inf') else None
        result['success'] = True
        
    except MemoryError:
        result['error'] = 'MemoryError: Insufficient memory'
    except Exception as e:
        result['error'] = f'Error: {str(e)}'
    
    return result


def find_graph_files(directory='.', pattern='*.npy'):
    """
    Find all graph files in a directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match (e.g., '*.npy', 'graph_*.npy')
    
    Returns:
        list: List of Path objects for found files
    """
    path = Path(directory)
    files = list(path.glob(pattern))
    
    # Also check for .npz files
    if pattern.endswith('.npy'):
        npz_pattern = pattern.replace('.npy', '.npz')
        files.extend(list(path.glob(npz_pattern)))
    
    return sorted(files)


def run_benchmarks(graph_files, output_csv='benchmark_results.csv', use_optimized=True):
    """
    Run benchmarks on multiple graph files and save results to CSV.
    
    Args:
        graph_files: List of file paths to graph files
        output_csv: Output CSV filename
        use_optimized: Whether to use the optimized algorithm
    
    Returns:
        list: List of benchmark results
    """
    results = []
    
    print("="*70)
    print("HELD-KARP ALGORITHM BENCHMARK")
    print("="*70)
    print(f"\nFound {len(graph_files)} graph file(s)")
    print(f"Output: {output_csv}")
    print(f"Algorithm: {'Optimized' if use_optimized else 'Standard'}")
    print("\n" + "-"*70)
    
    for i, graph_file in enumerate(graph_files, 1):
        graph_name = graph_file.stem  # Filename without extension
        
        print(f"\n[{i}/{len(graph_files)}] Processing: {graph_file.name}")
        
        try:
            # Load graph
            graph = load_graph(str(graph_file))
            
            # Validate graph
            if not validate_graph(graph):
                print(f"  [INVALID] Invalid graph format")
                results.append({
                    'graph_name': graph_name,
                    'size': None,
                    'states': None,
                    'memory_mb': None,
                    'time_seconds': None,
                    'min_cost': None,
                    'path_length': None,
                    'success': False,
                    'error': 'Invalid graph format'
                })
                continue
            
            n = len(graph)
            print(f"  Size: {n} nodes")
            print(f"  States: {2**n:,}")
            print(f"  Memory: {(2**n * n * 8) / (1024**2):.2f} MB")
            
            # Warn for large graphs
            if n > 20:
                print(f"  [WARNING] Size {n} may be too large!")
                response = input("  Continue? (y/n): ")
                if response.lower() != 'y':
                    print(f"  Skipped")
                    results.append({
                        'graph_name': graph_name,
                        'size': n,
                        'states': 2**n,
                        'memory_mb': (2**n * n * 8) / (1024**2),
                        'time_seconds': None,
                        'min_cost': None,
                        'path_length': None,
                        'success': False,
                        'error': 'Skipped by user'
                    })
                    continue
            
            # Run benchmark
            print(f"  Running benchmark...", end='', flush=True)
            result = benchmark_graph(graph, graph_name, use_optimized)
            
            if result['success']:
                print(f" [OK]")
                print(f"  Time: {result['time_seconds']:.6f} seconds")
                print(f"  Cost: {result['min_cost']}")
            else:
                print(f" [FAILED]")
                print(f"  Error: {result['error']}")
            
            results.append(result)
            
        except FileNotFoundError:
            print(f"  [ERROR] File not found")
            results.append({
                'graph_name': graph_name,
                'size': None,
                'states': None,
                'memory_mb': None,
                'time_seconds': None,
                'min_cost': None,
                'path_length': None,
                'success': False,
                'error': 'File not found'
            })
        except Exception as e:
            print(f"  [ERROR] Error loading: {e}")
            results.append({
                'graph_name': graph_name,
                'size': None,
                'states': None,
                'memory_mb': None,
                'time_seconds': None,
                'min_cost': None,
                'path_length': None,
                'success': False,
                'error': str(e)
            })
    
    print("\n" + "-"*70)
    
    # Write results to CSV
    write_results_to_csv(results, output_csv)
    
    # Print summary
    print_summary(results)
    
    return results


def write_results_to_csv(results, filename='benchmark_results.csv'):
    """
    Write benchmark results to a CSV file.
    
    Args:
        results: List of benchmark result dictionaries
        filename: Output CSV filename
    """
    if not results:
        print("\nNo results to write.")
        return
    
    # Define CSV columns
    fieldnames = [
        'graph_name',
        'size',
        'states',
        'memory_mb',
        'time_seconds',
        'min_cost',
        'path_length',
        'success',
        'error'
    ]
    
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n[OK] Results saved to: {filename}")

    except Exception as e:
        print(f"\n[ERROR] Error writing CSV: {e}")


def print_summary(results):
    """
    Print a summary of benchmark results.
    
    Args:
        results: List of benchmark result dictionaries
    """
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    # Summary table
    print(f"\n{'Graph':<20} {'Size':<8} {'Time (s)':<15} {'Cost':<12} {'Status':<10}")
    print("-"*70)
    
    for result in results:
        graph_name = result['graph_name'][:19]  # Truncate long names
        size = str(result['size']) if result['size'] else 'N/A'
        time_str = f"{result['time_seconds']:.6f}" if result['time_seconds'] else 'N/A'
        cost = str(result['min_cost']) if result['min_cost'] is not None else 'N/A'
        status = '[OK] Success' if result['success'] else '[FAILED]'
        
        print(f"{graph_name:<20} {size:<8} {time_str:<15} {cost:<12} {status:<10}")
    
    # Statistics
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print("\n" + "-"*70)
    print(f"Total graphs: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        times = [r['time_seconds'] for r in successful]
        print(f"\nExecution times:")
        print(f"  Fastest: {min(times):.6f} seconds")
        print(f"  Slowest: {max(times):.6f} seconds")
        print(f"  Average: {sum(times)/len(times):.6f} seconds")
    
    print("="*70)


def main():
    """
    Main entry point for the benchmark script.
    """
    # Configuration
    GRAPH_DIRECTORY = 'graph_structures_output'  # Directory containing graph files
    GRAPH_PATTERN = '*_adjacency.npy'  # Pattern to match graph files
    OUTPUT_CSV = 'benchmark_results.csv'
    USE_OPTIMIZED = True  # Use optimized version

    # You can override these with command-line arguments
    if len(sys.argv) > 1:
        GRAPH_DIRECTORY = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_CSV = sys.argv[2]
    
    # Find graph files
    graph_files = find_graph_files(GRAPH_DIRECTORY, GRAPH_PATTERN)
    
    if not graph_files:
        print(f"No graph files found in '{GRAPH_DIRECTORY}' matching pattern '{GRAPH_PATTERN}'")
        print("\nTip: Make sure your graph files are named like 'graph_4.npy', 'graph_8.npy', etc.")
        sys.exit(1)
    
    # Run benchmarks
    results = run_benchmarks(graph_files, OUTPUT_CSV, USE_OPTIMIZED)

    print(f"\n[OK] Benchmark complete!")
    print(f"  Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()