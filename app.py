import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import os
import glob
import json
import folium
import streamlit.components.v1 as components

from algorithms import (
    parse_tsp, compute_distance_matrix, get_candidate_neighbors,
    tour_cost, nearest_neighbor, multi_start_nearest_neighbor,
    fast_two_opt, fast_relocate, tabu_search, simulated_annealing,
    iterated_local_search
)

# Page Configuration
st.set_page_config(
    page_title="TSP Solver & Visualizer | Analytics Workshop",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-success { background-color: #DEF7EC; color: #03543F; }
    .badge-info { background-color: #E1EFFE; color: #1E429F; }
</style>
""", unsafe_allow_html=True)

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

@st.cache_data
def load_all_instances():
    files = sorted(glob.glob("tsp_data/*.tsp"))
    data = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        coords, meta = parse_tsp(f)
        data[name] = {'path': f, 'coords': coords, 'meta': meta}
    return data

instances = load_all_instances()

# Sidebar
st.sidebar.image("https://img.icons8.com/clouds/200/route.png", width=110)
st.sidebar.title("TSP Controls")

app_mode = st.sidebar.radio(
    "Navigation Mode",
    ["🏆 Benchmark Challenge (TSPLIB)", "🛺 Braving Bangalore Traffic", "📊 Hall of Fame Leaderboard", "🧠 Algorithm Docs & Moves"]
)

def plot_tsp_tour(coords, tour, title="TSP Tour", known_opt=None, current_cost=None):
    fig = go.Figure()
    
    # Tour lines
    if tour is not None and len(tour) > 0:
        tour_coords = coords[tour]
        closed_tour = np.vstack([tour_coords, tour_coords[0]])
        fig.add_trace(go.Scatter(
            x=closed_tour[:, 0],
            y=closed_tour[:, 1],
            mode='lines',
            line=dict(color='#2563EB', width=2),
            hoverinfo='none',
            name='Tour Route'
        ))
        
        # Start node
        fig.add_trace(go.Scatter(
            x=[tour_coords[0, 0]],
            y=[tour_coords[0, 1]],
            mode='markers+text',
            marker=dict(size=14, color='#EF4444', symbol='star'),
            name='Start / Home',
            text=["Start"],
            textposition="top center"
        ))
    
    # All city nodes
    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode='markers',
        marker=dict(size=6, color='#1E293B', line=dict(width=1, color='#FFFFFF')),
        hovertext=[f"City {i}: ({coords[i,0]:.1f}, {coords[i,1]:.1f})" for i in range(len(coords))],
        hoverinfo='text',
        name='Cities'
    ))
    
    subtitle = ""
    if current_cost is not None:
        subtitle += f"Distance: {current_cost:,} km"
    if known_opt is not None:
        subtitle += f" | Known Global Opt: {known_opt:,} km (Gap: {((current_cost-known_opt)/known_opt*100):.2f}%)"

    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:13px;color:gray'>{subtitle}</span>"),
        xaxis=dict(showgrid=True, zeroline=False),
        yaxis=dict(showgrid=True, zeroline=False),
        margin=dict(l=20, r=20, t=55, b=20),
        height=550,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ----------------- TAB 1: BENCHMARK CHALLENGE -----------------
if app_mode == "🏆 Benchmark Challenge (TSPLIB)":
    st.markdown('<div class="main-title">Traveling Salesman Problem Application</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Construct starting solutions via Nearest Neighbor and optimize using 2-Opt, Relocate, Tabu Search, and Iterated Local Search. Pure Python implementation with zero optimization libraries.</div>', unsafe_allow_html=True)

    col_side, col_main = st.columns([1, 3])
    
    with col_side:
        st.subheader("1. Instance Selection")
        selected_inst = st.selectbox(
            "Choose TSPLIB Dataset",
            list(instances.keys()),
            index=0
        )
        inst_info = instances[selected_inst]
        coords = inst_info['coords']
        n_cities = len(coords)
        known_opt = KNOWN_OPTIMA.get(selected_inst, None)
        
        st.info(f"**Cities (Nodes):** {n_cities}\n\n**Known Optimum:** {known_opt:,} km" if known_opt else f"Nodes: {n_cities}")
        
        st.subheader("2. Starting Heuristic")
        nn_mode = st.radio("Construct Starting Tour via", ["Nearest Neighbor (Start = 0)", "Multi-Start Nearest Neighbor"])
        
        st.subheader("3. Improvement Heuristics")
        algo_choice = st.selectbox(
            "Select Optimization Method",
            [
                "Iterated Local Search (ILS - Recommended)",
                "2-Opt Local Search",
                "Relocate / Or-Opt-1",
                "Tabu Search",
                "Simulated Annealing"
            ]
        )
        
        ils_time = st.slider("Max Search Time (seconds)", min_value=1.0, max_value=15.0, value=3.0, step=0.5)
        run_btn = st.button("🚀 Run Optimizer", type="primary", use_container_width=True)

    with col_main:
        # Distance matrix & Candidate list
        dist_mat = compute_distance_matrix(coords, round_euc=True)
        cands = get_candidate_neighbors(dist_mat, k=min(35, n_cities - 1))
        
        # State management
        if 'current_tour' not in st.session_state or st.session_state.get('curr_inst') != selected_inst:
            # Baseline NN
            t_nn = nearest_neighbor(dist_mat, start_node=0)
            st.session_state['nn_tour'] = t_nn
            st.session_state['nn_cost'] = tour_cost(t_nn, dist_mat)
            st.session_state['current_tour'] = list(t_nn)
            st.session_state['current_cost'] = st.session_state['nn_cost']
            st.session_state['curr_inst'] = selected_inst
            st.session_state['history'] = []
            
        if run_btn:
            with st.spinner("Optimizing route..."):
                t_start = time.time()
                # 1. Starting tour
                if nn_mode == "Multi-Start Nearest Neighbor":
                    start_tour = multi_start_nearest_neighbor(dist_mat, max_starts=min(25, n_cities))
                else:
                    start_tour = nearest_neighbor(dist_mat, start_node=0)
                    
                init_cost = tour_cost(start_tour, dist_mat)
                st.session_state['nn_tour'] = start_tour
                st.session_state['nn_cost'] = init_cost
                
                # 2. Improvement
                if algo_choice == "2-Opt Local Search":
                    best_t, best_c = fast_two_opt(start_tour, dist_mat, cands, max_iterations=600)
                    hist = [(0.0, init_cost), (time.time() - t_start, best_c)]
                elif algo_choice == "Relocate / Or-Opt-1":
                    t_2opt, _ = fast_two_opt(start_tour, dist_mat, cands, max_iterations=100)
                    best_t, best_c = fast_relocate(t_2opt, dist_mat, cands, max_iterations=300)
                    hist = [(0.0, init_cost), (time.time() - t_start, best_c)]
                elif algo_choice == "Tabu Search":
                    best_t, best_c, tabu_h = tabu_search(start_tour, dist_mat, cands, max_iterations=250, tabu_tenure=15)
                    hist = [(i * (ils_time / len(tabu_h)), c) for i, c in enumerate(tabu_h)]
                elif algo_choice == "Simulated Annealing":
                    best_t, best_c, sa_h = simulated_annealing(start_tour, dist_mat, max_time_sec=ils_time)
                    hist = [(i * (ils_time / len(sa_h)), c) for i, c in enumerate(sa_h)]
                else: # ILS
                    best_t, best_c, hist = iterated_local_search(start_tour, dist_mat, cands, max_time_sec=ils_time)
                    
                st.session_state['current_tour'] = best_t
                st.session_state['current_cost'] = best_c
                st.session_state['history'] = hist
                st.session_state['run_time'] = time.time() - t_start
                st.success(f"Optimization finished in {st.session_state['run_time']:.2f}s!")

        # Metric cards
        m1, m2, m3, m4 = st.columns(4)
        nn_c = st.session_state['nn_cost']
        curr_c = st.session_state['current_cost']
        imp = ((nn_c - curr_c) / nn_c * 100) if nn_c > 0 else 0
        gap_opt = ((curr_c - known_opt) / known_opt * 100) if known_opt else None

        with m1:
            st.metric("Initial NN Route", f"{nn_c:,} km")
        with m2:
            st.metric("Optimized Route", f"{curr_c:,} km", delta=f"-{imp:.1f}%", delta_color="inverse")
        with m3:
            st.metric("Known Optimum", f"{known_opt:,} km" if known_opt else "N/A")
        with m4:
            st.metric("Optimality Gap", f"{gap_opt:.2f}%" if gap_opt is not None else "N/A")

        # Visualization
        fig = plot_tsp_tour(
            coords,
            st.session_state['current_tour'],
            title=f"Instance: {selected_inst} ({n_cities} Cities)",
            known_opt=known_opt,
            current_cost=curr_c
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if st.session_state.get('history') and len(st.session_state['history']) > 1:
            with st.expander("📈 Convergence Curve (Cost vs Time)", expanded=False):
                h_data = np.array(st.session_state['history'])
                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(x=h_data[:, 0], y=h_data[:, 1], mode='lines+markers', line=dict(color='#10B981', width=2)))
                if known_opt:
                    fig_h.add_hline(y=known_opt, line_dash="dash", line_color="red", annotation_text=f"Known Optimum ({known_opt})")
                fig_h.update_layout(title="Convergence History", xaxis_title="Time (seconds)", yaxis_title="Tour Distance (km)", template='plotly_white')
                st.plotly_chart(fig_h, use_container_width=True)

# ----------------- TAB 2: BANGALORE TOY PROBLEM -----------------
elif app_mode == "🛺 Braving Bangalore Traffic":
    st.markdown('<div class="main-title">Case Study: Braving Bangalore Traffic (Rakhi Tour)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Recreating Slides 1–8: Visiting 7 cousins from Home and returning. Comparison of Nearest Neighbor, Local Search (Swap) trapped in local minimum, and Escapes via Relocation and Tabu Search.</div>', unsafe_allow_html=True)

    # Coordinates for the 8 nodes in Bangalore map layout
    node_coords = {
        'Home': np.array([2.0, 8.0]),
        'A': np.array([3.5, 6.5]),
        'B': np.array([6.0, 7.8]),
        'C': np.array([12.0, 7.2]),
        'D': np.array([9.0, 3.2]),
        'E': np.array([13.0, 2.2]),
        'F': np.array([7.0, 1.5]),
        'G': np.array([3.5, 3.0]),
    }

    # Road network edges with distances
    graph_edges = [
        ('Home', 'A', 1), ('Home', 'B', 2), ('Home', 'G', 5),
        ('A', 'B', 1), ('A', 'G', 2),
        ('B', 'C', 5), ('B', 'D', 3),
        ('C', 'D', 1), ('C', 'E', 3),
        ('D', 'E', 2), ('D', 'F', 2),
        ('E', 'F', 6),
        ('F', 'G', 2)
    ]

    scenario = st.radio(
        "Select Problem Step from Lecture Slides:",
        [
            "1. Nearest Cousin Next (Starting Solution) - 22 km",
            "2. Swap Move Trapped in Local Minimum - 20 km",
            "3. Escape 1: Relocate Move (Global Minimum) - 19 km",
            "4. Escape 2: Tabu Search (Uphill Move to 21km, then 19km) - 19 km"
        ]
    )

    if "1. Nearest" in scenario:
        route = ['Home', 'A', 'B', 'D', 'C', 'E', 'F', 'G', 'Home']
        dist = 22
        exp = "Greedily picking the closest cousin: H -> A (1) -> B (1) -> D (3) -> C (1) -> E (3) -> F (6) -> G (2) -> H (5) = 22 km. Leaves a terrible 6 km stretch from E to F!"
    elif "2. Swap Move" in scenario:
        route = ['Home', 'B', 'D', 'C', 'E', 'F', 'G', 'A', 'Home']
        dist = 20
        exp = "Swapping Home and A drops distance to 20 km. However, all other 26 possible swaps either produce invalid roads or worse routes. The search is trapped in a LOCAL MINIMUM!"
    elif "3. Escape 1" in scenario:
        route = ['Home', 'B', 'C', 'E', 'D', 'F', 'G', 'A', 'Home']
        dist = 19
        exp = "Change Neighborhood (Relocate): Take D out of route and visit D after E. Drops B->D (3), D->C (1), E->F (6) and adds B->C (5), E->D (2), D->F (2). The long 6 km E->F road is gone! Global Minimum: 19 km."
    else:
        route = ['Home', 'B', 'C', 'E', 'D', 'F', 'G', 'A', 'Home']
        dist = 19
        exp = "Tabu Search: Plain hill climbing refused swap C-D because it increased route to 21 km. Tabu search accepts this worse step, marks (C,D) as tabu, and on the very next step swaps (D,E) reaching 19 km!"

    # Real geographic GPS coordinates for Bangalore landmarks matching Slide 1
    bangalore_geo = {
        'Home': {'coords': [13.0115, 77.5348], 'label': '🏠 Home (Nandini Layout)'},
        'A':    {'coords': [12.9985, 77.5518], 'label': 'A: Malleshwaram / Subramanya Nagar'},
        'B':    {'coords': [12.9930, 77.5685], 'label': 'B: Rajajinagar Industrial Area'},
        'C':    {'coords': [12.9825, 77.6405], 'label': 'C: Bengaluru East / Indiranagar'},
        'D':    {'coords': [12.9680, 77.5950], 'label': 'D: Cubbon Park / Richmond Town'},
        'E':    {'coords': [12.9180, 77.6520], 'label': 'E: HSR Layout / Bellandur'},
        'F':    {'coords': [12.9320, 77.5840], 'label': 'F: Madhavan Park / Lalbagh / Jayanagar'},
        'G':    {'coords': [12.9620, 77.5280], 'label': 'G: Chandra Layout / Vijayanagar'},
    }

    map_view_type = st.radio(
        "Map Visualization Style:",
        ["🗺️ Real OpenStreetMap Geographic Map (Folium)", "📊 Schematic Lecture Diagram (Plotly)"],
        horizontal=True
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Route Sequence")
        st.code(" ➔ ".join(route), language="text")
        st.metric("Total Route Distance", f"{dist} km")
        st.info(exp)
        
    with c2:
        if "Real OpenStreetMap" in map_view_type:
            # Build interactive Folium map centered on Bengaluru
            b_map = folium.Map(
                location=[12.965, 77.590],
                zoom_start=12,
                tiles="CartoDB positron"
            )

            # Draw background road network (gray dashed lines)
            for u, v, w in graph_edges:
                pt1 = bangalore_geo[u]['coords']
                pt2 = bangalore_geo[v]['coords']
                folium.PolyLine(
                    locations=[pt1, pt2],
                    color="#94A3B8",
                    weight=2,
                    dash_array="5, 8",
                    opacity=0.6,
                    tooltip=f"Road {u} ↔ {v}: {w} km"
                ).add_to(b_map)

            # Draw active route path
            route_color = "#10B981" if dist == 19 else ("#F59E0B" if dist == 20 else "#EF4444")
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                pt1 = bangalore_geo[u]['coords']
                pt2 = bangalore_geo[v]['coords']
                folium.PolyLine(
                    locations=[pt1, pt2],
                    color=route_color,
                    weight=5,
                    opacity=0.9,
                    tooltip=f"Leg {i+1}: {u} → {v}"
                ).add_to(b_map)

            # Add markers for Home and Cousins
            for name, info in bangalore_geo.items():
                is_home = name == 'Home'
                icon_color = 'blue' if is_home else 'orange'
                icon_name = 'home' if is_home else 'user'
                folium.Marker(
                    location=info['coords'],
                    tooltip=f"<b>{info['label']}</b>",
                    popup=folium.Popup(f"<b>{name}</b><br>{info['label']}", max_width=200),
                    icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa')
                ).add_to(b_map)

            components.html(b_map._repr_html_(), height=500)
        else:
            # Plot Bangalore schematic graph
            fig_b = go.Figure()
            # background road network
            for u, v, w in graph_edges:
                cu, cv = node_coords[u], node_coords[v]
                fig_b.add_trace(go.Scatter(
                    x=[cu[0], cv[0]], y=[cu[1], cv[1]],
                    mode='lines+text',
                    line=dict(color='#CBD5E1', width=1.5, dash='dot'),
                    text=["", f"<b>{w}</b>"],
                    textposition="middle center",
                    hoverinfo='none',
                    showlegend=False
                ))
                
            # active route edges
            for i in range(len(route) - 1):
                u, v = route[i], route[i+1]
                cu, cv = node_coords[u], node_coords[v]
                fig_b.add_trace(go.Scatter(
                    x=[cu[0], cv[0]], y=[cu[1], cv[1]],
                    mode='lines',
                    line=dict(color='#DC2626' if dist > 19 else '#059669', width=4),
                    hoverinfo='none',
                    showlegend=False
                ))
                
            # Node markers
            for name, coord in node_coords.items():
                color = '#1E3A8A' if name == 'Home' else '#D97706'
                fig_b.add_trace(go.Scatter(
                    x=[coord[0]], y=[coord[1]],
                    mode='markers+text',
                    marker=dict(size=22, color=color, line=dict(color='#FFFFFF', width=2)),
                    text=[name],
                    textposition="middle center",
                    textfont=dict(color='white', size=11, family='Arial Black'),
                    hoverinfo='text',
                    name=name
                ))
                
            fig_b.update_layout(
                title="Bangalore Traffic Cousin Network",
                template='plotly_white',
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                height=480,
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False
            )
            st.plotly_chart(fig_b, use_container_width=True)

# ----------------- TAB 3: LEADERBOARD -----------------
elif app_mode == "📊 Hall of Fame Leaderboard":
    st.markdown('<div class="main-title">🏆 Hall of Fame Leaderboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Benchmark results for all 10 challenge instances from tinyurl.com/tsp-data. Computed strictly using custom pure Python heuristic algorithms.</div>', unsafe_allow_html=True)

    if os.path.exists("hall_of_fame_results.json"):
        with open("hall_of_fame_results.json") as f:
            hof_data = json.load(f)
            
        rows = []
        for k, v in hof_data.items():
            rows.append({
                "Instance": k,
                "Nodes": v['nodes'],
                "Nearest Neighbor (km)": f"{v['nn_km']:,}",
                "2-Opt (km)": f"{v['two_opt_km']:,}",
                "Our Best Route (km)": f"{v['best_km']:,}",
                "Known Global Opt (km)": f"{v['optimal_km']:,}",
                "Optimality Gap": v['gap']
            })
            
        st.dataframe(rows, use_container_width=True)
        st.caption("Note: All instances benchmarked according to TSPLIB EUC_2D integer standard.")
    else:
        st.warning("Run hall_of_fame_optimizer.py to populate the complete leaderboard.")

# ----------------- TAB 4: ALGORITHM DOCS -----------------
elif app_mode == "🧠 Algorithm Docs & Moves":
    st.markdown('<div class="main-title">Implemented TSP Optimization Moves</div>', unsafe_allow_html=True)
    st.markdown(r"""
    ### 1. Nearest Neighbor (NN) - Constructive Heuristic
    - **Concept:** Starting from a home node, greedily choose the closest unvisited node.
    - **Complexity:** $O(N^2)$
    - **Pros/Cons:** Fast to construct, but leaves long crisscrossing edges towards the end.

    ### 2. 2-Opt Local Search (Edge Inversion)
    - **Concept:** Cut two edges $(u, v)$ and $(w, x)$ and reconnect as $(u, w)$ and $(v, x)$ by reversing the path between $v$ and $w$.
    - **Move Condition:** Accept only if $d(u,w) + d(v,x) < d(u,v) + d(w,x)$.
    - **Candidate Lists:** To scale to 500+ nodes in fractions of a second, we restrict search to the $K$-nearest candidate neighbors.

    ### 3. Relocate Move (Or-Opt-1)
    - **Concept:** Extract a single node $v$ from between $(u, w)$ and insert it between $(x, y)$.
    - **Move Condition:** Accept only if net change $\Delta < 0$. Helps untangle nodes trapped inside tight loops.

    ### 4. Tabu Search
    - **Concept:** Allows uphill (worse) moves to escape local minima, while keeping recently reversed edges in a **tabu list** for $T$ iterations to prevent cycling.
    - **Aspiration Criterion:** A tabu move is allowed if it produces a tour better than the best-ever recorded solution.

    ### 5. Iterated Local Search (ILS) with Double Bridge Perturbation
    - **Concept:** When 2-opt gets stuck in a local minimum, apply a 4-opt "double bridge" move (cutting the tour into 4 pieces and crossing them). A double bridge move cannot be undone by a single 2-opt move, effectively jumping into a promising adjacent basin of attraction.
    """)
