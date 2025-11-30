// Ejecutar: mpirun -np 4 ./lns_mpi_omp xqf131.tsp

#include <mpi.h>
#include <omp.h>

#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <limits>
#include <unordered_set>

using namespace std;
using vi = vector<int>;
using vvi = vector<vi>;

// -------------------- TSPLIB reader + distance --------------------
struct City { double x,y; };

vector<City> readTSPLIB(const string &filename) {
    ifstream file(filename);
    string line;
    vector<City> cities;
    bool readingCoords = false;

    if (!file) {
        cerr << "No se pudo abrir " << filename << endl;
        return cities;
    }

    while (getline(file, line)) {
        if (line.find("NODE_COORD_SECTION") != string::npos) {
            readingCoords = true;
            continue;
        }
        if (readingCoords) {
            if (line.find("EOF") != string::npos) break;
            istringstream iss(line);
            int id;
            double x, y;
            if (iss >> id >> x >> y)
                cities.push_back({x, y});
        }
    }
    return cities;
}

// Matriz de adyacencia con los 50 puntos más cercanos 
vvi computeDistanceMatrix(const vector<City>& cities) {
    int n = (int)cities.size();
    vvi dist(n, vi(n,0));
    for (int i=0;i<n;i++){
        for (int j=0;j<n;j++){
            if (i==j) continue;
            double dx = cities[i].x - cities[j].x;
            double dy = cities[i].y - cities[j].y;
            dist[i][j] = (int)lround(sqrt(dx*dx + dy*dy));
        }
    }
    return dist;
}

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
                // reverse segment (i+1 .. j)
                reverse(tour.begin()+i+1, tour.begin()+j+1);
                return true;
            }
        }
    }
    return false;
}

void twoOpt(vi &tour, const vvi &adj) {
    while (twoOptImproveOnce(tour, adj)) { /*repetir hasta alcanzar el "optimo" local*/ }
}

// -------------------- "Destroy" (remover k nodos) --------------------
vi destroyRandom(const vi &tour, int remove_k, mt19937 &rng) {
    int n = tour.size();
    vi removed;
    removed.reserve(remove_k);
    unordered_set<int> sel;
    uniform_int_distribution<int> distIdx(0, n-1);

    // "eliminar" nodos radom
    while ((int)sel.size() < remove_k) {
        int pos = distIdx(rng);
        sel.insert(pos);
    }
    // construir solución con los nodos restantes
    vi rem;
    rem.reserve(n - remove_k);
    for (int i=0;i<n;i++){
        if (!sel.count(i)) rem.push_back(tour[i]);
        else removed.push_back(tour[i]);
    }
    // return vector: first rem size, then rem..., then removed...
    vi out;
    out.push_back((int)rem.size());
    out.insert(out.end(), rem.begin(), rem.end());
    out.insert(out.end(), removed.begin(), removed.end());
    return out;
}

// -------------------- Repair: insercion "greedy"  de los nodos removidos --------------------
void greedyReinsert(vi &partial, const vi &removed, const vvi &adj) {
    // partial: sequencia actual de nodos
    // removed: nodes a insertar
    for (int node : removed) {
        int bestPos = 0;
        int bestDelta = numeric_limits<int>::max();
        int m = partial.size();
        if (m == 0) {
            partial.push_back(node);
            continue;
        }
        for (int pos = 0; pos <= m; ++pos) {
            // insertion between pos-1 and pos (circular)
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
    int REMOVE_PERCENT = 20; // porcentaje
    int CANDIDATES = 0; // si 0 -> usar omp threads count
    int MIG_FREQ = 200; // migración cada MIG_FREQ iteraciones
    int SEED = 12345;
};

pair<vi,int> runLNSIsland(const vvi &adj, int rank, int size, const Params &P) {
    int n = adj.size();
    mt19937 rng(P.SEED + rank*97 + (int)chrono::high_resolution_clock::now().time_since_epoch().count()%10000);
    int remove_k = max(1, (n * P.REMOVE_PERCENT) / 100);

    // initial solution
    vi current = nearestNeighborSolution(adj, rank % n);
    twoOpt(current, adj);
    int current_cost = tourCost(current, adj);

    vi best = current;
    int best_cost = current_cost;

    int candidates_per_iter = P.CANDIDATES > 0 ? P.CANDIDATES : omp_get_max_threads();

    for (int iter=1; iter<=P.ITER; ++iter) {
        // generate several candidates in parallel and pick best
        vi local_best_candidate;
        int local_best_cost = numeric_limits<int>::max();

        #pragma omp parallel
        {
            mt19937 thr_rng = rng;
            // mix thread id into rng
            thr_rng.seed(thr_rng() ^ (omp_get_thread_num()*9137) ^ (iter*131));
            vi thr_best;
            int thr_best_cost = numeric_limits<int>::max();

            #pragma omp for schedule(dynamic)
            for (int c = 0; c < candidates_per_iter; ++c) {
                // 1) destroy
                vi out = destroyRandom(current, remove_k, thr_rng);
                int remsize = out[0];
                vi partial(out.begin()+1, out.begin()+1+remsize);
                vi removed(out.begin()+1+remsize, out.end());

                // shuffle removed to diversify
                shuffle(removed.begin(), removed.end(), thr_rng);

                // 2) repair greedy
                greedyReinsert(partial, removed, adj);

                // 3) local search
                twoOpt(partial, adj);

                // 4) evaluate
                int cost = tourCost(partial, adj);

                // 5) accept into thread-best if better
                if (cost < thr_best_cost) {
                    thr_best_cost = cost;
                    thr_best = partial;
                }
            } // end for candidates

            // merge thread bests into local best
            #pragma omp critical
            {
                if (thr_best_cost < local_best_cost) {
                    local_best_cost = thr_best_cost;
                    local_best_candidate = thr_best;
                }
            }
        } // end parallel

        // acceptance: if better, always accept; otherwise maybe accept with small probability (simulated annealing style)
        if (local_best_cost < current_cost) {
            current = local_best_candidate;
            current_cost = local_best_cost;
            if (current_cost < best_cost) {
                best = current;
                best_cost = current_cost;
            }
        } else {
            // small chance to escape local opt: accept if random < p
            double p_accept = 0.01; // small probability
            uniform_real_distribution<double> ud(0.0,1.0);
            if (ud(rng) < p_accept) {
                current = local_best_candidate;
                current_cost = local_best_cost;
            }
        }

        // migration / share best every MIG_FREQ iterations
        if (iter % P.MIG_FREQ == 0) {
            // gather best costs
            vector<int> costs(size);
            MPI_Allgather(&best_cost, 1, MPI_INT, costs.data(), 1, MPI_INT, MPI_COMM_WORLD);

            // find argmin
            int mincost = costs[0];
            int minrank = 0;
            for (int r=1;r<size;r++){
                if (costs[r] < mincost) { mincost = costs[r]; minrank = r; }
            }

            // broadcast tour from minrank
            vi recvtour(n);
            if (rank == minrank) {
                // ensure best has size n
                if ((int)best.size() != n) {
                    // if somehow not, fallback to current
                    best = current;
                }
            }
            // All processes call Bcast with root = minrank
            if (rank == minrank) {
                MPI_Bcast(best.data(), n, MPI_INT, minrank, MPI_COMM_WORLD);
            } else {
                MPI_Bcast(recvtour.data(), n, MPI_INT, minrank, MPI_COMM_WORLD);
                // adopt the received tour as current (diversify)
                current = recvtour;
                current_cost = tourCost(current, adj);
                // optionally run local search
                twoOpt(current, adj);
                current_cost = tourCost(current, adj);
                if (current_cost < best_cost) {
                    best = current;
                    best_cost = current_cost;
                }
            }
        }

        // optional: print progress by rank 0
        if (iter % (P.MIG_FREQ*2) == 0 && rank==0) {
            cout << "[iter " << iter << "] rank0 best = " << best_cost << endl;
        }
    } // end iterations

    return {best, best_cost};
}

void saveTourSVG(
    const vector<City> &cities,
    const vector<int> &tour,
    const string &filename = "./tsp_path.svg",
    int width = 800,
    int height = 800
) {
    if (cities.empty() || tour.empty()) {
        cerr << "No hay datos para dibujar." << endl;
        return;
    }
    int n = cities.size();
    if ((int)tour.size() != n) {
        cerr << "Advertencia: el tour no tiene tamaño n. Intentando usarlo de todas formas.\n";
    }

    // Obtener extremos
    double minX = 1e18, maxX = -1e18;
    double minY = 1e18, maxY = -1e18;

    for (auto &c : cities) {
        minX = min(minX, c.x);
        maxX = max(maxX, c.x);
        minY = min(minY, c.y);
        maxY = max(maxY, c.y);
    }

    double margin = 20;
    double dx = maxX - minX;
    double dy = maxY - minY;
    // evitar division por cero (caso degenerado)
    double scaleX = (dx > 0.0) ? (width - 2 * margin) / dx : 1.0;
    double scaleY = (dy > 0.0) ? (height - 2 * margin) / dy : 1.0;

    ofstream svg(filename);
    svg << "<svg xmlns='http://www.w3.org/2000/svg' width='" << width 
        << "' height='" << height << "'>\n";
    svg << "<rect width='100%' height='100%' fill='white'/>\n";

    // Dibuja líneas del tour (incluye cierre last->first)
    svg << "<g stroke='blue' stroke-width='2'>\n";
    for (int i = 0; i < (int)tour.size(); ++i) {
        int a = tour[i];
        int b = tour[(i+1) % tour.size()]; // cerrar
        double x1 = margin + (cities[a].x - minX) * scaleX;
        double y1 = margin + (cities[a].y - minY) * scaleY;
        double x2 = margin + (cities[b].x - minX) * scaleX;
        double y2 = margin + (cities[b].y - minY) * scaleY;
        svg << "<line x1='" << x1 << "' y1='" << y1 
            << "' x2='" << x2 << "' y2='" << y2 << "' />\n";
    }
    svg << "</g>\n";

    // Dibuja los puntos de las ciudades
    svg << "<g fill='red' stroke='black'>\n";
    for (int i = 0; i < n; i++) {
        double x = margin + (cities[i].x - minX) * scaleX;
        double y = margin + (cities[i].y - minY) * scaleY;
        svg << "<circle cx='" << x << "' cy='" << y << "' r='4'/>\n";
        svg << "<text x='" << (x + 5) << "' y='" << (y - 5) 
            << "' font-size='12'>" << i << "</text>\n";
    }
    svg << "</g>\n";
    svg << "</svg>\n";
    svg.close();

    cout << "SVG generado: " << filename << endl;
}


// -------------------- main --------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) cerr << "Uso: " << argv[0] << " instancia.tsp\n";
        MPI_Finalize();
        return 1;
    }

    string fname = argv[1];
    auto cities = readTSPLIB(fname);
    if (cities.empty()) {
        if (rank==0) cerr << "Error leyendo TSPLIB\n";
        MPI_Finalize();
        return 1;
    }

    vvi adj = computeDistanceMatrix(cities);
    int n = adj.size();
    if (rank==0) cout << "N ciudades = " << n << ", procesos = " << size << ", threads = " << omp_get_max_threads() << endl;

    // parámetros (ajustables)
    Params P;
    P.ITER = 2000;            // número de iteraciones por isla (ajusta según tiempo)
    P.REMOVE_PERCENT = 20;    // porcentaje de nodos a remover en cada destroy
    P.CANDIDATES = 0;         // 0 -> usa omp threads
    P.MIG_FREQ = 100;         // frecuencia migración
    P.SEED = 123456 + rank*100;

    auto start_time = chrono::high_resolution_clock::now();

    auto [best_tour_local, best_cost_local] = runLNSIsland(adj, rank, size, P);

    // Recolectar mejor global
    vector<int> all_costs(size);
    MPI_Gather(&best_cost_local, 1, MPI_INT, all_costs.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vi best_tour_global;
    int best_cost_global = numeric_limits<int>::max();
    int best_rank = 0;
    if (rank == 0) {
        for (int r=0;r<size;r++){
            if (all_costs[r] < best_cost_global) {
                best_cost_global = all_costs[r];
                best_rank = r;
            }
        }
    }
    // Broadcast best_rank and best_cost
    MPI_Bcast(&best_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&best_cost_global, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // root of broadcast is best_rank; but all must call Bcast with same root
    best_tour_global.assign(n, -1);
    if (rank == best_rank) {
        MPI_Bcast(best_tour_local.data(), n, MPI_INT, best_rank, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(best_tour_global.data(), n, MPI_INT, best_rank, MPI_COMM_WORLD);
    }
    if (rank != best_rank) best_tour_local = best_tour_global;

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = chrono::high_resolution_clock::now();
    double secs = chrono::duration<double>(end_time - start_time).count();

    if (rank == 0) {
        cout << "Mejor costo global: " << best_cost_global << " (rank " << best_rank << ")\n";
        cout << "Tiempo total (s): " << secs << "\n";
        cout << "Tour: ";
        for (int v : best_tour_local) cout << v << ' ';
        cout << best_tour_local[0] << '\n';
        saveTourSVG(cities, best_tour_local);
    }

    MPI_Finalize();
    return 0;
}