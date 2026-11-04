import glob
import os
import time
import numpy as np
from algorithms import (
    parse_tsp, compute_distance_matrix, get_candidate_neighbors,
    tour_cost, nearest_neighbor, fast_two_opt, fast_relocate,
    iterated_local_search, tabu_search, simulated_annealing
)

KNOWN_OPTIMA = {
    'a280': 2579,
    'd198': 15780,
    'd493': 35002,
    'fl417': 11861,
    'lin318': 42029,
    'pcb442': 50778,
    'pr152': 73682,
    'pr226': 80369,
    'pr439': 107217,
    'ts225': 126643,
}

def benchmark_instance(tsp_path, ils_time=3.0):
    inst_name = os.path.splitext(os.path.basename(tsp_path))[0]
    coords, meta = parse_tsp(tsp_path)
    n = len(coords)
    dist = compute_distance_matrix(coords, round_euc=True)
    cands = get_candidate_neighbors(dist, k=30)
    
    # 1. Nearest Neighbor
    t0 = time.time()
    tour_nn = nearest_neighbor(dist, start_node=0)
    cost_nn = tour_cost(tour_nn, dist)
    time_nn = time.time() - t0
    
    # 2. 2-Opt
    t0 = time.time()
    tour_2opt, cost_2opt = fast_two_opt(tour_nn, dist, cands, max_iterations=400)
    time_2opt = time.time() - t0
    
    # 3. Relocate
    t0 = time.time()
    tour_reloc, cost_reloc = fast_relocate(tour_2opt, dist, cands, max_iterations=200)
    time_reloc = time.time() - t0
    
    # 4. ILS (Iterated Local Search with Double Bridge)
    t0 = time.time()
    tour_ils, cost_ils, _ = iterated_local_search(tour_reloc, dist, cands, max_time_sec=ils_time, max_iter=80)
    time_ils = time.time() - t0
    
    opt = KNOWN_OPTIMA.get(inst_name, None)
    gap = f"{((cost_ils - opt) / opt * 100):.2f}%" if opt else "N/A"
    
    return {
        'instance': inst_name,
        'n': n,
        'nn_cost': cost_nn,
        'two_opt_cost': cost_2opt,
        'reloc_cost': cost_reloc,
        'best_cost': cost_ils,
        'optimal': opt,
        'gap': gap
    }

if __name__ == '__main__':
    tsp_files = sorted(glob.glob('tsp_data/*.tsp'))
    print(f"{'Instance':10} | {'Nodes':5} | {'NN (km)':10} | {'2-Opt (km)':10} | {'Reloc (km)':10} | {'Best ILS (km)':13} | {'Known Opt':10} | {'Gap %':7}", flush=True)
    print("-" * 92, flush=True)
    results = []
    for f in tsp_files:
        res = benchmark_instance(f, ils_time=2.5)
        results.append(res)
        print(f"{res['instance']:10} | {res['n']:5} | {res['nn_cost']:10} | {res['two_opt_cost']:10} | {res['reloc_cost']:10} | {res['best_cost']:13} | {str(res['optimal']):10} | {res['gap']:7}", flush=True)
