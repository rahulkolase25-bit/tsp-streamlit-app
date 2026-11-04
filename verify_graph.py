import itertools

# Graph from slides
# Nodes: H, A, B, C, D, E, F, G
edges = {
    ('H', 'A'): 1,
    ('H', 'B'): 2,
    ('H', 'G'): 5,
    ('A', 'B'): 1,
    ('A', 'G'): 2,
    ('B', 'C'): 5,
    ('B', 'D'): 3,
    ('C', 'D'): 1,
    ('C', 'E'): 3,
    ('D', 'E'): 2,
    ('D', 'F'): 2,
    ('E', 'F'): 6,
    ('F', 'G'): 2,
}

# Undirected graph
dist = {}
for (u, v), w in edges.items():
    dist[(u, v)] = w
    dist[(v, u)] = w

nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
best_cost = float('inf')
best_tours = []

for perm in itertools.permutations(nodes):
    tour = ['H'] + list(perm) + ['H']
    cost = 0
    valid = True
    for i in range(len(tour) - 1):
        u, v = tour[i], tour[i+1]
        if (u, v) in dist:
            cost += dist[(u, v)]
        else:
            valid = False
            break
    if valid:
        if cost < best_cost:
            best_cost = cost
            best_tours = [(tour, cost)]
        elif cost == best_cost:
            best_tours.append((tour, cost))

print(f"Optimal cost: {best_cost} km")
print(f"Number of optimal tours: {len(best_tours)}")
for t, c in best_tours:
    print(" -> ".join(t), "=", c)

