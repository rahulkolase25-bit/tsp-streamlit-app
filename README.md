# Traveling Salesman Problem (TSP) Interactive Solver & Visualizer

A pure-Python TSP optimization application and visualizer inspired by [tspvis.com](https://tspvis.com), built for the **AI for Analytics Workshop Challenge**.

---

## 🌟 Key Highlights
- **Strictly Zero Optimization Libraries**: All heuristics and metaheuristics (Nearest Neighbor, 2-Opt, Relocate, Tabu Search, Simulated Annealing, Iterated Local Search) are implemented in pure Python and NumPy from scratch.
- **Preloaded Benchmark Instances**: Includes all 10 TSPLIB benchmark instances from `https://tinyurl.com/tsp-data`.
- **Bangalore Traffic Case Study**: Dedicated interactive tab recreating the classroom slides (Nearest Neighbor 22 km → Swap local minimum 20 km → Relocate/Tabu global minimum 19 km).
- **Streamlit Interactive UI**: Dynamic Plotly tour path rendering, convergence curve graphs, and real-time metric cards.

---

## 🏆 Hall of Fame Benchmark Results

Benchmarked strictly under the standard TSPLIB `EUC_2D` integer rounding metric:

| Instance | Nodes | Nearest Neighbor (Baseline) | 2-Opt Local Search | Our Best Route (km) | Known Global Optimum | Optimality Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ts225** | 225 | 152,493 km | 130,412 km | **126,643 km** | 126,643 km | **0.00% (Exact Opt)** |
| **pr152** | 152 | 85,699 km | 77,617 km | **73,682 km** | 73,682 km | **0.00% (Exact Opt)** |
| **d198** | 198 | 18,240 km | 16,821 km | **15,876 km** | 15,780 km | **0.61%** |
| **pr226** | 226 | 94,683 km | 84,920 km | **81,016 km** | 80,369 km | **0.81%** |
| **a280** | 280 | 3,157 km | 2,778 km | **2,626 km** | 2,579 km | **1.82%** |
| **pcb442** | 442 | 61,979 km | 53,844 km | **52,110 km** | 50,778 km | **2.62%** |
| **lin318** | 318 | 54,019 km | 45,527 km | **43,387 km** | 42,029 km | **3.23%** |
| **d493** | 493 | 41,660 km | 36,864 km | **36,516 km** | 35,002 km | **4.33%** |
| **fl417** | 417 | 15,013 km | 12,801 km | **12,380 km** | 11,861 km | **4.38%** |
| **pr439** | 439 | 131,281 km | 115,202 km | **114,419 km** | 107,217 km | **6.72%** |

---

## 🚀 How to Run Locally

1. Make sure Python 3.10+ is available.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the application:
   ```bash
   streamlit run app.py
   ```
4. Open your browser at `http://localhost:8501`.

---

## 🌐 Deploying to Streamlit Cloud (share.streamlit.io)

1. Push this folder to a GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "TSP Streamlit Application"
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
2. Log into [https://share.streamlit.io](https://share.streamlit.io).
3. Click **"New App"**, select your repository, set the main file path to `app.py`, and click **Deploy**.

