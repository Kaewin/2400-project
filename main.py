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
from brute_force import brute_force, brute_force_optimized
from monte_carlo import monte_carlo


def benchmark_graph(graph, graph_name="unknown", use_optimized=True, run_brute_force=True):
    """
    Benchmark Held-Karp, Monte Carlo, and (optionally) Brute Force algorithms on a single graph.

    Args:
        graph: Adjacency matrix (numpy array)
        graph_name: Name identifier for the graph
        use_optimized: Whether to use the optimized versions
        run_brute_force: Whether to run brute force (disabled for large graphs)

    Returns:
        dict: Benchmark results containing:
            - graph_name: Name of the graph
            - size: Number of nodes
            - states: Number of DP states (2^n)
            - memory_mb: Estimated memory usage in MB

            - hk_time_seconds: Held-Karp execution time
            - hk_min_cost: Held-Karp minimum cost

            - bf_time_seconds: Brute force execution time (if run)
            - bf_min_cost: Brute force minimum cost (if run)
            - costs_match: Whether both exact algorithms found same cost
            - speedup: How much faster Held-Karp is vs Brute Force

            - mc_time_seconds: Monte Carlo execution time
            - mc_min_cost: Monte Carlo minimum cost found
            - mc_iterations: Number of Monte Carlo iterations
            - mc_gap_vs_opt: Relative gap (mc - hk) / hk if both available

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

        # exact algorithms
        'hk_time_seconds': None,
        'hk_min_cost': None,
        'bf_time_seconds': None,
        'bf_min_cost': None,
        'costs_match': None,
        'speedup': None,

        # Monte Carlo heuristic
        'mc_time_seconds': None,
        'mc_min_cost': None,
        'mc_iterations': None,
        'mc_gap_vs_opt': None,

        'path_length': None,
        'success': False,
        'error': None
    }

    try:
        # Run Held-Karp algorithm
        hk_algorithm = held_karp_optimized if use_optimized else held_karp

        start_time = time.perf_counter()
        hk_min_cost = hk_algorithm(graph)
        end_time = time.perf_counter()
        hk_time = end_time - start_time

        result['hk_time_seconds'] = hk_time
        result['hk_min_cost'] = hk_min_cost if hk_min_cost != float('inf') else None

        # --- Monte Carlo heuristic benchmark ---
        if n <= 10:
            mc_iterations = 20_000
        elif n <= 15:
            mc_iterations = 10_000
        elif n <= 20:
            mc_iterations = 5_000
        else:
            mc_iterations = 2_000

        start_time = time.perf_counter()
        mc_min_cost = monte_carlo(graph, iterations=mc_iterations)
        end_time = time.perf_counter()
        mc_time = end_time - start_time

        result['mc_time_seconds'] = mc_time
        result['mc_min_cost'] = mc_min_cost if mc_min_cost != float('inf') else None
        result['mc_iterations'] = mc_iterations

        if result['hk_min_cost'] is not None and result['mc_min_cost'] is not None:
            result['mc_gap_vs_opt'] = (
                (result['mc_min_cost'] - result['hk_min_cost']) / result['hk_min_cost']
            )

        # Run Brute Force algorithm (only for small graphs)
        if run_brute_force and n <= 10:  # Brute force gets very slow after n=10
            bf_algorithm = brute_force_optimized if use_optimized else brute_force

            start_time = time.perf_counter()
            bf_min_cost = bf_algorithm(graph)
            end_time = time.perf_counter()
            bf_time = end_time - start_time

            result['bf_time_seconds'] = bf_time
            result['bf_min_cost'] = bf_min_cost if bf_min_cost != float('inf') else None

            # Compare results
            if result['hk_min_cost'] is not None and result['bf_min_cost'] is not None:
                result['costs_match'] = (result['hk_min_cost'] == result['bf_min_cost'])
                result['speedup'] = bf_time / hk_time if hk_time > 0 else None

        result['path_length'] = n + 1 if hk_min_cost != float('inf') else None
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
                    'hk_time_seconds': None,
                    'hk_min_cost': None,
                    'bf_time_seconds': None,
                    'bf_min_cost': None,
                    'costs_match': None,
                    'speedup': None,
                    'mc_time_seconds': None,
                    'mc_min_cost': None,
                    'mc_iterations': None,
                    'mc_gap_vs_opt': None,
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
                        'hk_time_seconds': None,
                        'hk_min_cost': None,
                        'bf_time_seconds': None,
                        'bf_min_cost': None,
                        'costs_match': None,
                        'speedup': None,
                        'mc_time_seconds': None,
                        'mc_min_cost': None,
                        'mc_iterations': None,
                        'mc_gap_vs_opt': None,
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
                print(f"  Held-Karp: {result['hk_time_seconds']:.6f}s, Cost: {result['hk_min_cost']}")

                if result.get('mc_time_seconds') is not None:
                    if result.get('mc_gap_vs_opt') is not None:
                        gap_pct = result['mc_gap_vs_opt'] * 100
                        gap_str = f"{gap_pct:.2f}%"
                    else:
                        gap_str = "N/A"
                    print(
                        f"  Monte Carlo: {result['mc_time_seconds']:.6f}s, "
                        f"Cost: {result['mc_min_cost']}, "
                        f"Iters: {result['mc_iterations']}, Gap vs HK: {gap_str}"
                    )

                if result['bf_time_seconds'] is not None:
                    print(f"  Brute Force: {result['bf_time_seconds']:.6f}s, Cost: {result['bf_min_cost']}")
                    print(f"  Match: {result['costs_match']}, Speedup: {result['speedup']:.2f}x")
                else:
                    print(f"  Brute Force: Skipped (graph too large)")
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
                'hk_time_seconds': None,
                'hk_min_cost': None,
                'bf_time_seconds': None,
                'bf_min_cost': None,
                'costs_match': None,
                'speedup': None,
                'mc_time_seconds': None,
                'mc_min_cost': None,
                'mc_iterations': None,
                'mc_gap_vs_opt': None,
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
                'hk_time_seconds': None,
                'hk_min_cost': None,
                'bf_time_seconds': None,
                'bf_min_cost': None,
                'costs_match': None,
                'speedup': None,
                'mc_time_seconds': None,
                'mc_min_cost': None,
                'mc_iterations': None,
                'mc_gap_vs_opt': None,
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
        'hk_time_seconds',
        'hk_min_cost',
        'bf_time_seconds',
        'bf_min_cost',
        'costs_match',
        'speedup',
        'mc_time_seconds',
        'mc_min_cost',
        'mc_iterations',
        'mc_gap_vs_opt',
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
    print("\n" + "="*110)
    print("BENCHMARK SUMMARY")
    print("="*110)

    # Summary table
    print(
        f"\n{'Graph':<20} {'Size':<6} "
        f"{'HK Time':<12} {'BF Time':<12} "
        f"{'MC Time':<12} {'MC Gap%':<10} "
        f"{'Speedup':<10} {'Match':<8} {'Status':<10}"
    )
    print("-"*110)

    for result in results:
        graph_name = result['graph_name'][:19]  # Truncate long names
        size = str(result['size']) if result['size'] else 'N/A'
        hk_time = f"{result['hk_time_seconds']:.6f}s" if result.get('hk_time_seconds') else 'N/A'
        bf_time = (
            f"{result['bf_time_seconds']:.6f}s"
            if result.get('bf_time_seconds') is not None
            else 'Skipped'
        )
        mc_time = (
            f"{result['mc_time_seconds']:.6f}s"
            if result.get('mc_time_seconds') is not None
            else 'N/A'
        )
        if result.get('mc_gap_vs_opt') is not None:
            mc_gap_pct = f"{result['mc_gap_vs_opt'] * 100:.2f}%"
        else:
            mc_gap_pct = 'N/A'

        speedup = f"{result['speedup']:.2f}x" if result.get('speedup') else 'N/A'
        match = str(result['costs_match']) if result.get('costs_match') is not None else 'N/A'
        status = '[OK]' if result['success'] else '[FAIL]'

        print(
            f"{graph_name:<20} {size:<6} "
            f"{hk_time:<12} {bf_time:<12} "
            f"{mc_time:<12} {mc_gap_pct:<10} "
            f"{speedup:<10} {match:<8} {status:<10}"
        )

    # Statistics
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    with_brute_force = [r for r in successful if r['bf_time_seconds'] is not None]

    print("\n" + "-"*110)
    print(f"Total graphs: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        hk_times = [r['hk_time_seconds'] for r in successful if r['hk_time_seconds']]
        print(f"\nHeld-Karp execution times:")
        print(f"  Fastest: {min(hk_times):.6f} seconds")
        print(f"  Slowest: {max(hk_times):.6f} seconds")
        print(f"  Average: {sum(hk_times)/len(hk_times):.6f} seconds")

    if with_brute_force:
        bf_times = [r['bf_time_seconds'] for r in with_brute_force]
        speedups = [r['speedup'] for r in with_brute_force if r['speedup']]
        all_match = all(r['costs_match'] for r in with_brute_force if r['costs_match'] is not None)

        print(f"\nBrute Force comparison ({len(with_brute_force)} graphs):")
        print(f"  All costs match: {all_match}")
        print(f"  Average speedup: {sum(speedups)/len(speedups):.2f}x")
        print(f"  Max speedup: {max(speedups):.2f}x")

    print("="*110)


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
