"""
High-Optimization Runner to get Top-3 Hall of Fame numbers for all 10 TSPLIB instances.
Runs Multi-Start Nearest Neighbor + 2-Opt + Relocate + Multi-temperature ILS.
"""

import glob
import os
import time
import json
import numpy as np
from algorithms import (
    parse_tsp, compute_distance_matrix, get_candidate_neighbors,
    tour_cost, nearest_neighbor, multi_start_nearest_neighbor,
    fast_two_opt, fast_relocate, iterated_local_search, double_bridge_move
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

def deep_optimize(tsp_path, time_limit_sec=10.0):
    inst_name = os.path.splitext(os.path.basename(tsp_path))[0]
    coords, meta = parse_tsp(tsp_path)
    n = len(coords)
    dist = compute_distance_matrix(coords, round_euc=True)
    cands = get_candidate_neighbors(dist, k=35)
    
    # 1. Standard Nearest Neighbor from start node 0 (as required baseline by assignment)
    t_start = time.time()
    tour_nn0 = nearest_neighbor(dist, start_node=0)
    cost_nn0 = tour_cost(tour_nn0, dist)
    
    # Baseline 2-opt from NN0
    tour_2opt0, cost_2opt0 = fast_two_opt(tour_nn0, dist, cands, max_iterations=400)
    
    # Multi-start NN to find best initial basin of attraction
    num_starts = min(20, n)
    tour_ms = multi_start_nearest_neighbor(dist, max_starts=num_starts)
    cost_ms = tour_cost(tour_ms, dist)
    
    # Best starting tour
    best_tour, best_cost = fast_two_opt(tour_ms, dist, cands, max_iterations=500)
    best_tour, best_cost = fast_relocate(best_tour, dist, cands, max_iterations=250)
    
    curr_tour = list(best_tour)
    curr_cost = best_cost
    
    # Iterated Local Search with adaptive kicks
    it = 0
    while time.time() - t_start < time_limit_sec:
        it += 1
        # Double bridge kick
        pert = double_bridge_move(curr_tour)
        cand, cand_cost = fast_two_opt(pert, dist, cands, max_iterations=200)
        cand, cand_cost = fast_relocate(cand, dist, cands, max_iterations=100)
        
        # Acceptance
        if cand_cost < curr_cost:
            curr_tour = cand
            curr_cost = cand_cost
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_tour = list(curr_tour)
        elif np.random.rand() < 0.05: # occasional escape kick
            curr_tour = cand
            curr_cost = cand_cost
            
    opt = KNOWN_OPTIMA.get(inst_name, None)
    gap = f"{((best_cost - opt) / opt * 100):.2f}%" if opt else "N/A"
    
    return {
        'instance': inst_name,
        'nodes': n,
        'nn_km': cost_nn0,
        'two_opt_km': cost_2opt0,
        'best_km': best_cost,
        'optimal_km': opt,
        'gap': gap,
        'tour': best_tour
    }

if __name__ == '__main__':
    tsp_files = sorted(glob.glob('tsp_data/*.tsp'))
    print(f"{'Instance':10} | {'Nodes':5} | {'NN (km)':10} | {'2-Opt (km)':10} | {'Optimized (km)':15} | {'Known Opt':10} | {'Gap %':7}", flush=True)
    print("-" * 94, flush=True)
    
    all_results = {}
    for f in tsp_files:
        res = deep_optimize(f, time_limit_sec=6.0)
        all_results[res['instance']] = {
            'nodes': res['nodes'],
            'nn_km': res['nn_km'],
            'two_opt_km': res['two_opt_km'],
            'best_km': res['best_km'],
            'optimal_km': res['optimal_km'],
            'gap': res['gap']
        }
        print(f"{res['instance']:10} | {res['nodes']:5} | {res['nn_km']:10} | {res['two_opt_km']:10} | {res['best_km']:15} | {str(res['optimal_km']):10} | {res['gap']:7}", flush=True)
        # Save best tour
        np.save(f"tsp_data/{res['instance']}_best_tour.npy", np.array(res['tour']))

    with open('hall_of_fame_results.json', 'w') as out_f:
        json.dump(all_results, out_f, indent=2)
    print("\nAll results saved to hall_of_fame_results.json!")

