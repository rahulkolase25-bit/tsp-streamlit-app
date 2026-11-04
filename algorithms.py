"""
High-Performance Pure-Python & NumPy TSP Optimization Suite
Implements:
1. Nearest Neighbor (Single-start and Multi-start)
2. 2-Opt Local Search (with fast Candidate Lists / full scan)
3. Relocate / Or-Opt-1 Local Search
4. Tabu Search
5. Simulated Annealing
6. Iterated Local Search (ILS) with 4-opt Double-Bridge perturbation

Strictly NO external optimization libraries used (no OR-Tools, no scipy.optimize).
"""

import math
import random
import time
import numpy as np

def parse_tsp(filepath):
    """Parses a TSPLIB .tsp file and extracts coordinates and metadata."""
    coords = []
    meta = {}
    with open(filepath, 'r') as f:
        reading_coords = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('NODE_COORD_SECTION'):
                reading_coords = True
                continue
            if line.startswith('EOF'):
                break
            if reading_coords:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        coords.append((x, y))
                    except ValueError:
                        continue
            else:
                if ':' in line:
                    k, v = line.split(':', 1)
                    meta[k.strip()] = v.strip()
    return np.array(coords, dtype=np.float64), meta

def compute_distance_matrix(coords, round_euc=True):
    """
    Computes Euclidean distance matrix.
    TSPLIB standard EUC_2D specifies integer rounding: nint(sqrt(dx^2 + dy^2)).
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.hypot(diff[:, :, 0], diff[:, :, 1])
    if round_euc:
        return np.round(dist).astype(np.int64)
    return dist

def get_candidate_neighbors(dist_matrix, k=25):
    """Precomputes k nearest neighbors for each node for fast local search."""
    n = len(dist_matrix)
    k = min(k, n - 1)
    # argsort along axis 1, excluding self
    cand = np.argsort(dist_matrix, axis=1)[:, 1:k+1]
    return cand

def tour_cost(tour, dist_matrix):
    """Calculates total cost of a cyclic tour."""
    n = len(tour)
    cost = 0
    for i in range(n):
        cost += dist_matrix[tour[i], tour[(i + 1) % n]]
    return int(cost)

def nearest_neighbor(dist_matrix, start_node=0):
    """
    Constructive Heuristic: Nearest Neighbor.
    Starts at start_node, greedily visits nearest unvisited node next.
    """
    n = len(dist_matrix)
    visited = np.zeros(n, dtype=bool)
    tour = [start_node]
    visited[start_node] = True
    curr = start_node
    
    for _ in range(n - 1):
        # fast row mask
        row = dist_matrix[curr].copy()
        row[visited] = 10**12 # mask visited
        next_node = int(np.argmin(row))
        tour.append(next_node)
        visited[next_node] = True
        curr = next_node
        
    return tour

def multi_start_nearest_neighbor(dist_matrix, max_starts=30):
    """Runs Nearest Neighbor starting from multiple nodes and picks best."""
    n = len(dist_matrix)
    num_starts = min(n, max_starts)
    starts = random.sample(range(n), num_starts) if num_starts < n else range(n)
    best_tour = None
    best_cost = float('inf')
    
    for s in starts:
        t = nearest_neighbor(dist_matrix, start_node=s)
        c = tour_cost(t, dist_matrix)
        if c < best_cost:
            best_cost = c
            best_tour = t
    return best_tour

def fast_two_opt(initial_tour, dist_matrix, cand_neighbors=None, max_iterations=500):
    """
    Fast 2-Opt Local Search using Candidate Neighbor Lists (or full scan if cand is None).
    """
    tour = list(initial_tour)
    n = len(tour)
    cost = tour_cost(tour, dist_matrix)
    
    pos = {node: i for i, node in enumerate(tour)}
    improved = True
    it = 0
    
    while improved and it < max_iterations:
        improved = False
        it += 1
        for i in range(n):
            u = tour[i]
            v = tour[(i + 1) % n]
            d_uv = dist_matrix[u, v]
            
            # Check candidate neighbors of u
            targets = cand_neighbors[u] if cand_neighbors is not None else [tour[j] for j in range(n)]
            
            for w in targets:
                if w == u or w == v:
                    continue
                j = pos[w]
                # we want segment from i+1 to j
                if j == (i - 1 + n) % n or j == (i + 1) % n or j == i:
                    continue
                # edge (w, x)
                x = tour[(j + 1) % n]
                if x == u or x == v:
                    continue
                
                # Check 2-opt delta
                # If we cut (u, v) and (w, x), new edges are (u, w) and (v, x)
                if i < j:
                    delta = (dist_matrix[u, w] + dist_matrix[v, x]) - (d_uv + dist_matrix[w, x])
                    if delta < 0:
                        tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                        cost += delta
                        for p in range(i + 1, j + 1):
                            pos[tour[p]] = p
                        improved = True
                        break
                elif j < i:
                    delta = (dist_matrix[x, v] + dist_matrix[w, u]) - (dist_matrix[w, x] + d_uv)
                    if delta < 0:
                        tour[j + 1 : i + 1] = reversed(tour[j + 1 : i + 1])
                        cost += delta
                        for p in range(j + 1, i + 1):
                            pos[tour[p]] = p
                        improved = True
                        break
            if improved:
                break
                
    return tour, int(cost)

def fast_relocate(initial_tour, dist_matrix, cand_neighbors=None, max_iterations=300):
    """
    Relocate Move (Or-Opt 1):
    Removes city v and inserts it between x and y if it reduces length.
    """
    tour = list(initial_tour)
    n = len(tour)
    cost = tour_cost(tour, dist_matrix)
    pos = {node: i for i, node in enumerate(tour)}
    
    improved = True
    it = 0
    while improved and it < max_iterations:
        improved = False
        it += 1
        for i in range(n):
            v = tour[i]
            u = tour[(i - 1 + n) % n]
            w = tour[(i + 1) % n]
            saved = dist_matrix[u, v] + dist_matrix[v, w] - dist_matrix[u, w]
            
            targets = cand_neighbors[v] if cand_neighbors is not None else [tour[j] for j in range(n)]
            for x in targets:
                if x == v or x == u or x == w:
                    continue
                j = pos[x]
                y = tour[(j + 1) % n]
                added = dist_matrix[x, v] + dist_matrix[v, y] - dist_matrix[x, y]
                delta = added - saved
                if delta < 0:
                    node = tour.pop(i)
                    insert_pos = j if j < i else j - 1
                    tour.insert(insert_pos + 1, node)
                    cost += delta
                    # Update pos map
                    pos = {n_id: idx for idx, n_id in enumerate(tour)}
                    improved = True
                    break
            if improved:
                break
                
    return tour, int(cost)

def double_bridge_move(tour):
    """4-opt Double-bridge perturbation."""
    n = len(tour)
    if n < 8:
        t = list(tour)
        random.shuffle(t)
        return t
    cut = sorted(random.sample(range(1, n), 3))
    i, j, k = cut[0], cut[1], cut[2]
    A = tour[:i]
    B = tour[i:j]
    C = tour[j:k]
    D = tour[k:]
    return A + D + C + B

def iterated_local_search(initial_tour, dist_matrix, cand_neighbors=None, max_time_sec=3.0, max_iter=100):
    """
    Iterated Local Search (ILS):
    Applies double-bridge perturbation + local search to continuously improve solutions.
    """
    curr_tour, curr_cost = fast_two_opt(initial_tour, dist_matrix, cand_neighbors, max_iterations=200)
    curr_tour, curr_cost = fast_relocate(curr_tour, dist_matrix, cand_neighbors, max_iterations=100)
    
    best_tour, best_cost = list(curr_tour), curr_cost
    t0 = time.time()
    it = 0
    history = [(0.0, best_cost)]
    
    while (time.time() - t0) < max_time_sec and it < max_iter:
        it += 1
        pert = double_bridge_move(curr_tour)
        cand, cand_cost = fast_two_opt(pert, dist_matrix, cand_neighbors, max_iterations=150)
        cand, cand_cost = fast_relocate(cand, dist_matrix, cand_neighbors, max_iterations=60)
        
        # Acceptance criterion
        if cand_cost < curr_cost or random.random() < 0.03:
            curr_tour = cand
            curr_cost = cand_cost
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_tour = list(curr_tour)
                history.append((round(time.time() - t0, 3), best_cost))
                
    return best_tour, best_cost, history

def tabu_search(initial_tour, dist_matrix, cand_neighbors=None, max_iterations=300, tabu_tenure=15, sample_k=30):
    """
    Tabu Search with 2-opt moves.
    Maintains a tabu list of recently altered edges to avoid cycles.
    """
    current_tour = list(initial_tour)
    best_tour = list(initial_tour)
    current_cost = tour_cost(current_tour, dist_matrix)
    best_cost = current_cost
    
    n = len(current_tour)
    tabu_dict = {}
    history = [best_cost]
    
    for it in range(1, max_iterations + 1):
        best_cand = None
        best_cand_delta = float('inf')
        best_cand_edge = None
        
        sample_nodes = random.sample(range(n), min(sample_k, n))
        for i in sample_nodes:
            u = current_tour[i]
            v = current_tour[(i + 1) % n]
            d_uv = dist_matrix[u, v]
            
            targets = cand_neighbors[u] if cand_neighbors is not None else random.sample(range(n), min(sample_k, n))
            for w in targets:
                if w == u or w == v:
                    continue
                try:
                    j = current_tour.index(w)
                except ValueError:
                    continue
                if j == (i - 1 + n) % n or j == (i + 1) % n or j == i:
                    continue
                x = current_tour[(j + 1) % n]
                if x == u or x == v:
                    continue
                
                if i < j:
                    delta = (dist_matrix[u, w] + dist_matrix[v, x]) - (d_uv + dist_matrix[w, x])
                    edge_key = (min(u, w), max(u, w))
                    is_tabu = tabu_dict.get(edge_key, 0) > it
                    if not is_tabu or (current_cost + delta < best_cost):
                        if delta < best_cand_delta:
                            best_cand_delta = delta
                            best_cand = (i, j)
                            best_cand_edge = edge_key
                            
        if best_cand is not None:
            i, j = best_cand
            current_tour[i + 1 : j + 1] = reversed(current_tour[i + 1 : j + 1])
            current_cost += best_cand_delta
            tabu_dict[best_cand_edge] = it + tabu_tenure
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = list(current_tour)
        history.append(best_cost)
        
    return best_tour, best_cost, history

def simulated_annealing(initial_tour, dist_matrix, max_time_sec=3.0, initial_temp=200.0, cooling_rate=0.999):
    """Simulated Annealing metaheuristic."""
    tour = list(initial_tour)
    cost = tour_cost(tour, dist_matrix)
    best_tour = list(tour)
    best_cost = cost
    
    n = len(tour)
    T = initial_temp
    t0 = time.time()
    history = [best_cost]
    step = 0
    
    while (time.time() - t0) < max_time_sec and T > 1e-4:
        step += 1
        i = random.randint(0, n - 3)
        j = random.randint(i + 2, n - 1 if i > 0 else n - 2)
        
        a, b = tour[i], tour[i + 1]
        c, d = tour[j], tour[(j + 1) % n]
        delta = (dist_matrix[a, c] + dist_matrix[b, d]) - (dist_matrix[a, b] + dist_matrix[c, d])
        
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
            tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
            cost += delta
            if cost < best_cost:
                best_cost = cost
                best_tour = list(tour)
        T *= cooling_rate
        if step % 200 == 0:
            history.append(best_cost)
            
    return best_tour, best_cost, history
