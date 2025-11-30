#include <mpi.h>
#include <omp.h>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include "utils/tools.cpp"

using namespace std;
using vi = vector<int>;
using vvi = vector<vi>;


// Variable Global para métricas
long long local_evaluations = 0; // "Nodos" o Candidatos evaluados

// -------------------- Utilities (cost, nearest neighbor) --------------------
int tourCost(const vi &tour, const vvi &adj) {
    int n = (int)tour.size();
    int cost = 0;
    for (int i=0;i<n-1;i++) cost += adj[tour[i]][tour[i+1]];
    cost += adj[tour[n-1]][tour[0]];
    return cost;
}

vi nearestNeighborSolution(const vvi &adj, int start=0) {
    int n = adj.size();
    vi tour;
    tour.reserve(n);
    vector<char> used(n,false);
    int cur = start;
    tour.push_back(cur);
    used[cur] = true;
    for (int k=1; k<n; k++){
        int best = -1;
        int bestd = numeric_limits<int>::max();
        for (int j=0; j<n; j++){
            if (!used[j] && adj[cur][j] < bestd) {
                bestd = adj[cur][j];
                best = j;
            }
        }
        if (best==-1) break;
        tour.push_back(best);
        used[best]=true;
        cur = best;
    }
    return tour;
}

// -------------------- 2-opt local search --------------------
bool twoOptImproveOnce(vi &tour, const vvi &adj) {
    int n = tour.size();
    for (int i = 0; i < n-1; ++i) {
        int a = tour[i], b = tour[(i+1)%n];
        for (int j = i+2; j < n; ++j) {
            if (j == i) continue;
            int c = tour[j], d = tour[(j+1)%n];
            int delta = adj[a][c] + adj[b][d] - adj[a][b] - adj[c][d];
            if (delta < 0) {
                reverse(tour.begin()+i+1, tour.begin()+j+1);
                return true;
            }
        }
    }
    return false;
}

void twoOpt(vi &tour, const vvi &adj) {
    while (twoOptImproveOnce(tour, adj)) { /* repetir */ }
}

// -------------------- "Destroy" (remover k nodos) --------------------
vi destroyRandom(const vi &tour, int remove_k, mt19937 &rng) {
    int n = tour.size();
    vi removed;
    removed.reserve(remove_k);
    unordered_set<int> sel;
    uniform_int_distribution<int> distIdx(0, n-1);

    while ((int)sel.size() < remove_k) {
        int pos = distIdx(rng);
        sel.insert(pos);
    }
    vi rem;
    rem.reserve(n - remove_k);
    for (int i=0;i<n;i++){
        if (!sel.count(i)) rem.push_back(tour[i]);
        else removed.push_back(tour[i]);
    }
    vi out;
    out.push_back((int)rem.size());
    out.insert(out.end(), rem.begin(), rem.end());
    out.insert(out.end(), removed.begin(), removed.end());
    return out;
}

// -------------------- Repair: insercion "greedy" --------------------
void greedyReinsert(vi &partial, const vi &removed, const vvi &adj) {
    for (int node : removed) {
        int bestPos = 0;
        int bestDelta = numeric_limits<int>::max();
        int m = partial.size();
        if (m == 0) {
            partial.push_back(node);
            continue;
        }
        for (int pos = 0; pos <= m; ++pos) {
            int left = partial[(pos-1+m)%m];
            int right = partial[pos % m];
            int delta = adj[left][node] + adj[node][right] - adj[left][right];
            if (delta < bestDelta) {
                bestDelta = delta;
                bestPos = pos;
            }
        }
        partial.insert(partial.begin() + bestPos, node);
    }
}

// -------------------- LNS per-process loop --------------------
struct Params {
    int ITER = 5000;
    int REMOVE_PERCENT = 20; 
    int CANDIDATES = 0; 
    int MIG_FREQ = 200; 
    int SEED = 12345;
};

pair<vi,int> runLNSIsland(const vvi &adj, int rank, int size, const Params &P) {
    int n = adj.size();
    mt19937 rng(P.SEED + rank*97 + (int)chrono::high_resolution_clock::now().time_since_epoch().count()%10000);
    int remove_k = max(1, (n * P.REMOVE_PERCENT) / 100);

    vi current = nearestNeighborSolution(adj, rank % n);
    twoOpt(current, adj);
    int current_cost = tourCost(current, adj);

    vi best = current;
    int best_cost = current_cost;

    int candidates_per_iter = P.CANDIDATES > 0 ? P.CANDIDATES : omp_get_max_threads();

    for (int iter=1; iter<=P.ITER; ++iter) {
        vi local_best_candidate;
        int local_best_cost = numeric_limits<int>::max();

        // Contamos cuántos candidatos vamos a procesar en esta iteración
        // Lo hacemos atómico o simplemente sumamos al final del bloque parallel
        long long iter_evals = 0;

        #pragma omp parallel
        {
            mt19937 thr_rng = rng;
            thr_rng.seed(thr_rng() ^ (omp_get_thread_num()*9137) ^ (iter*131));
            vi thr_best;
            int thr_best_cost = numeric_limits<int>::max();
            long long thr_evals = 0;

            #pragma omp for schedule(dynamic)
            for (int c = 0; c < candidates_per_iter; ++c) {
                vi out = destroyRandom(current, remove_k, thr_rng);
                int remsize = out[0];
                vi partial(out.begin()+1, out.begin()+1+remsize);
                vi removed(out.begin()+1+remsize, out.end());

                shuffle(removed.begin(), removed.end(), thr_rng);
                greedyReinsert(partial, removed, adj);
                twoOpt(partial, adj); // O(N^2) heavy work

                int cost = tourCost(partial, adj);
                if (cost < thr_best_cost) {
                    thr_best_cost = cost;
                    thr_best = partial;
                }
                thr_evals++; // Contamos 1 evaluación completa (Destroy+Repair+2Opt)
            }

            #pragma omp critical
            {
                if (thr_best_cost < local_best_cost) {
                    local_best_cost = thr_best_cost;
                    local_best_candidate = thr_best;
                }
                iter_evals += thr_evals;
            }
        } 
        
        // Acumular al global local
        local_evaluations += iter_evals;

        if (local_best_cost < current_cost) {
            current = local_best_candidate;
            current_cost = local_best_cost;
            if (current_cost < best_cost) {
                best = current;
                best_cost = current_cost;
            }
        } else {
            double p_accept = 0.01;
            uniform_real_distribution<double> ud(0.0,1.0);
            if (ud(rng) < p_accept) {
                current = local_best_candidate;
                current_cost = local_best_cost;
            }
        }

        // Migration logic
        if (iter % P.MIG_FREQ == 0) {
            vector<int> costs(size);
            MPI_Allgather(&best_cost, 1, MPI_INT, costs.data(), 1, MPI_INT, MPI_COMM_WORLD);
            int mincost = costs[0];
            int minrank = 0;
            for (int r=1;r<size;r++){
                if (costs[r] < mincost) { mincost = costs[r]; minrank = r; }
            }

            vi recvtour(n);
            if (rank == minrank) {
                if ((int)best.size() != n) best = current;
            }
            if (rank == minrank) {
                MPI_Bcast(best.data(), n, MPI_INT, minrank, MPI_COMM_WORLD);
            } else {
                MPI_Bcast(recvtour.data(), n, MPI_INT, minrank, MPI_COMM_WORLD);
                current = recvtour;
                current_cost = tourCost(current, adj);
                if (current_cost < best_cost) {
                    best = current;
                    best_cost = current_cost;
                }
            }
        }
    }

    return {best, best_cost};
}

void saveTourSVG(const vector<City> &cities, const vector<int> &tour, const string &filename = "./tsp_path.svg") {
    // Función de guardado SVG (mantenida igual para visualización)
    if (cities.empty() || tour.empty()) return;
    int n = cities.size();
    ofstream svg(filename);
    svg << "<svg xmlns='http://www.w3.org/2000/svg' width='800' height='800'>\n";
    svg << "<rect width='100%' height='100%' fill='white'/>\n";
    
    double minX = 1e18, maxX = -1e18, minY = 1e18, maxY = -1e18;
    for (auto &c : cities) { minX = min(minX, c.x); maxX = max(maxX, c.x); minY = min(minY, c.y); maxY = max(maxY, c.y); }
    double scaleX = 760.0 / (maxX - minX);
    double scaleY = 760.0 / (maxY - minY);

    svg << "<g stroke='blue' stroke-width='2'>\n";
    for (int i = 0; i < n; ++i) {
        int a = tour[i], b = tour[(i+1)%n];
        svg << "<line x1='" << 20+(cities[a].x-minX)*scaleX << "' y1='" << 20+(cities[a].y-minY)*scaleY 
            << "' x2='" << 20+(cities[b].x-minX)*scaleX << "' y2='" << 20+(cities[b].y-minY)*scaleY << "' />\n";
    }
    svg << "</g>\n</svg>\n";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int number_of_nodes = 0; // Dinámico

    if (rank == 0) {
        // Manejo de argumentos: ./lns_mpi xqf131.tsp [N]
        if (argc < 2) {
            cerr << "Uso: " << argv[0] << " instancia.tsp [N_puntos]\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (argc >= 3) {
            number_of_nodes = atoi(argv[2]);
        }
    }
    
    // Compartir N con todos
    MPI_Bcast(&number_of_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    string fname = (argc >= 2) ? argv[1] : "data/xqf131.tsp";
    auto cities = readTSPLIB(fname);
    
    if (cities.empty()) {
        if (rank==0) cerr << "Error leyendo " << fname << "\n";
        MPI_Finalize();
        return 1;
    }

    // Recortar dataset si se especificó N > 0
    if (number_of_nodes > 0 && number_of_nodes < (int)cities.size()) {
        cities.resize(number_of_nodes);
    }
    int n = cities.size(); // N real final

    vvi adj = computeDistanceMatrix(cities);
    
    // Params
    Params P;
    P.ITER = 2000; 
    P.REMOVE_PERCENT = 20; 
    P.CANDIDATES = 0; 
    P.MIG_FREQ = 100;
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = chrono::high_resolution_clock::now();

    auto [best_tour_local, best_cost_local] = runLNSIsland(adj, rank, size, P);

    // Reducción de métricas
    long long total_evals_global = 0;
    MPI_Reduce(&local_evaluations, &total_evals_global, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Recolectar mejor costo
    vector<int> all_costs(size);
    MPI_Gather(&best_cost_local, 1, MPI_INT, all_costs.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    int best_cost_global = numeric_limits<int>::max();
    int best_rank = 0;
    if (rank == 0) {
        for (int r=0;r<size;r++) {
            if (all_costs[r] < best_cost_global) {
                best_cost_global = all_costs[r];
                best_rank = r;
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = chrono::high_resolution_clock::now();
    double secs = chrono::duration<double>(end_time - start_time).count();

    if (rank == 0) {
        double estimated_flops = (double)total_evals_global * n * n;
        double gflops = (secs > 0) ? (estimated_flops / secs / 1e9) : 0.0;

        cout << "\n=== RESULTADOS LNS HEURÍSTICO ===" << endl;
        cout << "N (Size): " << n << endl;
        cout << "Procesos MPI: " << size << " | Threads OMP: " << omp_get_max_threads() << endl;
        cout << "Mejor Costo: " << best_cost_global << endl;
        cout << "Tiempo (s): " << secs << endl;
        cout << "Tiempo (ms): " << secs * 1000.0 << endl;
        cout << "Candidatos Evaluados (Total): " << total_evals_global << endl;
        cout << "FLOPs Estimados: " << (long long)estimated_flops << endl;
        cout << "GFLOPs/s: " << gflops << endl;
        cout << "=================================\n" << endl;
        
        saveTourSVG(cities, best_tour_local); 
    }

    MPI_Finalize();
    return 0;
}


