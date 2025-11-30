#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <limits>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <atomic> 
#include <omp.h>

using namespace std;

const int INF = INT_MAX;

struct Point {
    int id;
    double x, y;
};

// --- VARIABLES GLOBALES ---
int final_res = INF;
vector<int> final_path;
int N_GLOBAL = 0;
const int MAX_TASK_DEPTH = 10; 

// --- MÉTRICAS ---
std::atomic<long long> nodes_visited{0}; 

void copyToFinal(const vector<int>& curr_path) {
    for (int i = 0; i < N_GLOBAL; i++)
        final_path[i] = curr_path[i];
    final_path[N_GLOBAL] = curr_path[0];
}

int firstMin(const vector<vector<int>>& adj, int i) {
    int min = INF;
    for (int k = 0; k < N_GLOBAL; k++)
        if (adj[i][k] < min && i != k)
            min = adj[i][k];
    return min;
}

int secondMin(const vector<vector<int>>& adj, int i) {
    int first = INF, second = INF;
    for (int j = 0; j < N_GLOBAL; j++) {
        if (i == j) continue;
        if (adj[i][j] <= first) {
            second = first;
            first = adj[i][j];
        } else if (adj[i][j] <= second && adj[i][j] != first) {
            second = adj[i][j];
        }
    }
    return second;
}

void TSPRec(const vector<vector<int>>& adj, int curr_bound, int curr_weight, 
            int level, vector<int> curr_path, vector<bool> visited) {
    
    // Contamos este nodo como visitado/procesado
    nodes_visited++;

    if (curr_weight >= final_res) return;

    if (level == N_GLOBAL) {
        if (adj[curr_path[level - 1]][curr_path[0]] != 0) {
            int curr_res = curr_weight + adj[curr_path[level - 1]][curr_path[0]];
            if (curr_res < final_res) {
                #pragma omp critical
                {
                    if (curr_res < final_res) {
                        copyToFinal(curr_path);
                        final_res = curr_res;
                    }
                }
            }
        }
        return;
    }

    for (int i = 0; i < N_GLOBAL; i++) {
        if (adj[curr_path[level - 1]][i] != 0 && !visited[i]) {
            int temp = curr_bound;
            int new_weight = curr_weight + adj[curr_path[level - 1]][i];

            if (level == 1)
                curr_bound -= ((firstMin(adj, curr_path[level - 1]) + firstMin(adj, i)) / 2);
            else
                curr_bound -= ((secondMin(adj, curr_path[level - 1]) + firstMin(adj, i)) / 2);

            if (curr_bound + new_weight < final_res) {
                vector<int> next_path = curr_path;
                next_path[level] = i;
                vector<bool> next_visited = visited;
                next_visited[i] = true;

                if (level < MAX_TASK_DEPTH) {
                    #pragma omp task shared(adj, final_res) firstprivate(curr_bound, new_weight, level, next_path, next_visited)
                    {
                        TSPRec(adj, curr_bound, new_weight, level + 1, next_path, next_visited);
                    }
                } else {
                    TSPRec(adj, curr_bound, new_weight, level + 1, next_path, next_visited);
                }
            }
            curr_bound = temp;
        }
    }
}

void TSP(const vector<vector<int>>& adj) {
    vector<int> curr_path(N_GLOBAL + 1);
    vector<bool> visited(N_GLOBAL, false);
    int curr_bound = 0;
    std::fill(curr_path.begin(), curr_path.end(), -1);

    for (int i = 0; i < N_GLOBAL; i++)
        curr_bound += (firstMin(adj, i) + secondMin(adj, i));
    curr_bound = (curr_bound & 1) ? curr_bound / 2 + 1 : curr_bound / 2;

    visited[0] = true;
    curr_path[0] = 0;

    #pragma omp parallel
    {
        #pragma omp single
        {
            // Resetear contador
            nodes_visited = 0; 
            cout << ">> Hilos activos: " << omp_get_num_threads() << endl;
            TSPRec(adj, curr_bound, 0, 1, curr_path, visited);
        }
    }
}

int calculateDistance(Point p1, Point p2) {
    return round(sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2)));
}

int main() {
    string filename = "data/xqf131.txt"; 
    ifstream file(filename);
    if (!file.is_open()) { cerr << "Error: falta xqf131.txt" << endl; return 1; }

    vector<Point> points;
    int id; double x, y;
    while (file >> id >> x >> y) points.push_back({id, x, y});
    file.close();

    cout << "Puntos totales: " << points.size() << endl;
    cout << "Ingrese N (Tamaño del problema): ";
    cin >> N_GLOBAL;

    if (N_GLOBAL > points.size()) N_GLOBAL = points.size();
    final_path.resize(N_GLOBAL + 1);
    vector<vector<int>> adj(N_GLOBAL, vector<int>(N_GLOBAL));
    
    for (int i = 0; i < N_GLOBAL; i++) {
        for (int j = 0; j < N_GLOBAL; j++) {
            if (i == j) adj[i][j] = 0;
            else adj[i][j] = calculateDistance(points[i], points[j]);
        }
    }

    auto start = chrono::high_resolution_clock::now();
    TSP(adj);
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);


    long long total_nodes = nodes_visited.load();
    double time_sec = duration.count() / 1000.0;
    
    long long estimated_flops = total_nodes * N_GLOBAL * 2; 
    double flops_per_sec = (time_sec > 0) ? estimated_flops / time_sec : 0;

    cout << "\nCosto Mínimo: " << final_res << endl;
    //cout << "Ruta: ";
    //for (int i = 0; i < N_GLOBAL; i++) {
    //    cout << points[final_path[i]].id << (i + 1 < N_GLOBAL ? " -> " : "");
    //}
    //cout << " -> " << points[final_path[0]].id << endl;

    cout << "\nTiempo ms" << endl;
    cout << "N (Tamaño): " << N_GLOBAL << endl;
    cout << "Hilos (P): " << omp_get_max_threads() << endl; 
    cout << "Tiempo (ms): " << duration.count() << endl;
    cout << "Nodos Visitados: " << total_nodes << endl;
    cout << "FLOPs Totales (Estimado): " << estimated_flops << endl;
    cout << "GFLOPs/s (aprox): " << flops_per_sec / 1e9 << endl;

    return 0;
}