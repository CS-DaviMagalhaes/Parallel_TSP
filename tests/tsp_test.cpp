/*
    TSP Branch and Bound con EXPORTACIÓN DE DATOS.
    
    1. Lee xqf131.txt
    2. Calcula la ruta más corta.
    3. Genera 'solution.txt' para que Python lo grafique.
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <limits>
#include <iomanip>
#include <algorithm>
#include <cstring>

using namespace std;

const int INF = INT_MAX;

struct Point {
    int id;
    double x, y;
};

int final_res = INF;
vector<int> final_path;
vector<bool> visited;
int N_GLOBAL = 0;

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

void TSPRec(const vector<vector<int>>& adj, int curr_bound, int curr_weight, int level, vector<int>& curr_path) {
    if (level == N_GLOBAL) {
        if (adj[curr_path[level - 1]][curr_path[0]] != 0) {
            int curr_res = curr_weight + adj[curr_path[level - 1]][curr_path[0]];
            if (curr_res < final_res) {
                copyToFinal(curr_path);
                final_res = curr_res;
            }
        }
        return;
    }

    for (int i = 0; i < N_GLOBAL; i++) {
        if (adj[curr_path[level - 1]][i] != 0 && !visited[i]) {
            int temp = curr_bound;
            curr_weight += adj[curr_path[level - 1]][i];

            if (level == 1)
                curr_bound -= ((firstMin(adj, curr_path[level - 1]) + firstMin(adj, i)) / 2);
            else
                curr_bound -= ((secondMin(adj, curr_path[level - 1]) + firstMin(adj, i)) / 2);

            if (curr_bound + curr_weight < final_res) {
                curr_path[level] = i;
                visited[i] = true;
                TSPRec(adj, curr_bound, curr_weight, level + 1, curr_path);
            }

            curr_weight -= adj[curr_path[level - 1]][i];
            curr_bound = temp;
            std::fill(visited.begin(), visited.end(), false);
            for (int j = 0; j <= level - 1; j++)
                visited[curr_path[j]] = true;
        }
    }
}

void TSP(const vector<vector<int>>& adj) {
    vector<int> curr_path(N_GLOBAL + 1);
    int curr_bound = 0;
    std::fill(visited.begin(), visited.end(), false);
    std::fill(curr_path.begin(), curr_path.end(), -1);

    for (int i = 0; i < N_GLOBAL; i++)
        curr_bound += (firstMin(adj, i) + secondMin(adj, i));

    curr_bound = (curr_bound & 1) ? curr_bound / 2 + 1 : curr_bound / 2;
    visited[0] = true;
    curr_path[0] = 0;
    TSPRec(adj, curr_bound, 0, 1, curr_path);
}

int calculateDistance(Point p1, Point p2) {
    return round(sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2)));
}

// --- NUEVA FUNCIÓN PARA EXPORTAR DATOS ---
void saveSolutionForPlotting(const vector<Point>& points) {
    ofstream out("solution.txt");
    if (!out.is_open()) {
        cerr << "Error al crear solution.txt" << endl;
        return;
    }
    
    // Escribir encabezado para referencia (opcional)
    // Formato: ID X Y
    for (int i = 0; i <= N_GLOBAL; i++) {
        int idx = final_path[i];
        out << points[idx].id << " " << points[idx].x << " " << points[idx].y << endl;
    }
    out.close();
    cout << "\n[INFO] Datos guardados en 'solution.txt' para graficar." << endl;
}

int main() {
    string filename = "data/xqf131.txt"; 
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: No se encuentra xqf131.txt" << endl;
        return 1;
    }

    vector<Point> points;
    int id;
    double x, y;

    while (file >> id >> x >> y) {
        points.push_back({id, x, y});
    }
    file.close();

    cout << "Puntos cargados: " << points.size() << endl;
    cout << "Ingrese N (ej. 10, 12, 15): ";
    cin >> N_GLOBAL;

    if (N_GLOBAL > points.size()) N_GLOBAL = points.size();
    if (N_GLOBAL < 2) return 0;

    if (N_GLOBAL > 20) {
        cout << "ADVERTENCIA: N alto. Puede tardar mucho." << endl;
    }

    final_path.resize(N_GLOBAL + 1);
    visited.resize(N_GLOBAL);

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

    //cout << "Costo Minimo: " << final_res << endl;
    cout << "Tiempo: " << duration.count() << " ms" << endl;

    // Exportar para Python
    saveSolutionForPlotting(points);

    return 0;
}