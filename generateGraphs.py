# --- Graph Generator: Complete, Lattice, Checkerboard Alternating ---
# Generates K4/K8/K16, lattice graphs, and checkerboard alternating graphs.
# Saves adjacency matrices as .npy and images as .png.

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "graph_structures_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Helper functions ----------

def save_graph_plot(G, filename, title=None, node_size=300):
    """Draws and saves a plot of the graph."""
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=node_size)
    plt.title(title if title else f"Graph (n={len(G.nodes())})")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_adjacency(G, name_prefix):
    """Convert graph to adjacency matrix and save as .npy."""
    A = nx.to_numpy_array(G, dtype=int)
    npy_path = os.path.join(OUTPUT_DIR, f"{name_prefix}_adjacency.npy")
    np.save(npy_path, A)
    print(f"Saved: {npy_path}  (shape={A.shape})")
    return A

# ---------- 1. Complete Graphs ----------
def generate_complete_graph(n):
    G = nx.complete_graph(n)
    save_adjacency(G, f"K{n}")
    save_graph_plot(G, os.path.join(OUTPUT_DIR, f"K{n}.png"), title=f"Complete Graph K{n}")
    return G

# ---------- 2. Lattice (Grid-like) Graphs ----------
def generate_lattice_graph(n):
    """Create lattice (grid) graph with limited connectivity."""
    if n == 4:
        dims = (2, 2)
    elif n == 8:
        dims = (2, 4)
    elif n == 16:
        dims = (4, 4)
    else:
        raise ValueError("Lattice only defined for n=4,8,16 in this script.")

    G = nx.grid_2d_graph(*dims)
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    save_adjacency(G, f"Lattice{n}")
    save_graph_plot(G, os.path.join(OUTPUT_DIR, f"Lattice{n}.png"),
                    title=f"Lattice Graph ({dims[0]}x{dims[1]})")
    return G

# ---------- 3. Checkerboard Alternating Graphs ----------
def generate_checkerboard_graph(n):
    """
    Create a checkerboard-like alternating pattern graph.
    Concept:
      - Nodes are colored 'black' or 'white' based on parity.
      - Connect each black node to nearby white nodes (no black-black or white-white).
      - Produces a bipartite, alternating pattern.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Determine black/white sets (like alternating tiles)
    black_nodes = [i for i in range(n) if i % 2 == 0]
    white_nodes = [i for i in range(n) if i % 2 == 1]

    # Connect black ↔ white in a local alternating pattern
    for b in black_nodes:
        # Connect each black node to two nearby white nodes
        for offset in [1, 3]:  # "checker" offsets
            w = (b + offset) % n
            if w in white_nodes:
                G.add_edge(b, w)

    save_adjacency(G, f"Checker{n}")
    save_graph_plot(G, os.path.join(OUTPUT_DIR, f"Checker{n}.png"),
                    title=f"Checkerboard Graph (n={n})")
    return G

# ---------- Run All ----------
sizes = [4, 8, 16]
for n in sizes:
    print(f"\n=== Generating graphs for n={n} ===")
    generate_complete_graph(n)
    generate_lattice_graph(n)
    generate_checkerboard_graph(n)

print("\n✅ All graphs (Complete, Lattice, Checkerboard) saved in:", OUTPUT_DIR)
