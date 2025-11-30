#include <iostream>
#include <vector>
#include <mpi.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits.h>
#include <iomanip>
#include <cstdlib> 

using namespace std;


int number_of_nodes = 10; 
long long local_nodes_visited = 0;

struct City {
    double x, y;
};

// Función de lectura
vector<City> readTSPLIB(const string &filename) {
    ifstream file(filename);
    string line;
    vector<City> cities;
    bool readingCoords = false;

    if (!file.is_open()) return cities;

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

vector<vector<int>> computeDistanceMatrix(const vector<City>& cities) {
    int n = cities.size();
    vector<vector<int>> dist(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            double dx = cities[i].x - cities[j].x;
            double dy = cities[i].y - cities[j].y;
            dist[i][j] = (int)round(sqrt(dx*dx + dy*dy));
        }
    }
    return dist;
}

vector<int> final_path; // Final solution (path of salesman)
vector<bool> visited; // Tracks visited nodes in a particular path
int final_res = INT_MAX; // Stores the final minimum weight of shortest tour.

// Function to copy temporary solution to the final solution
void copyToFinal(const vector<int>& curr_path)
{
    int N = curr_path.size() - 1;
    final_path.resize(N + 1);
    for (int i=0; i<N; i++)
        final_path[i] = curr_path[i];
    final_path[N] = curr_path[0];
}
// Function to find the minimum edge cost having an end at the vertex i
int firstMin(const vector<vector<int>>& adj, int i)
{
    int min = INT_MAX;
    int N = adj.size();
    for (int k=0; k<N; k++)
        if (adj[i][k]<min && i != k)
            min = adj[i][k];
    return min;
}
// function to find the second minimum edge cost having an end at the vertex i
int secondMin(const vector<vector<int>>& adj, int i)
{
    int first = INT_MAX, second = INT_MAX;
    int N = adj.size();
    for (int j=0; j<N; j++)
    {
        if (i == j) continue;
        if (adj[i][j] <= first) {
            second = first;
            first = adj[i][j];
        }
        else if (adj[i][j] <= second && adj[i][j] != first)
            second = adj[i][j];
    }
    return second;
}
// function that takes as arguments:
// curr_bound -> lower bound of the root node
// curr_weight-> stores the weight of the path so far
// level-> current level while moving in the search space tree
// curr_path[] -> where the solution is being stored which
//                would later be copied to final_path[]
// my_rank -> MPI rank of the current process
// n_procs -> Total number of MPI processes
void TSPRec(const vector<vector<int>>& adj, int curr_bound, int curr_weight, int level, vector<int>& curr_path, int my_rank, int n_procs)
{
    // Contar nodo visitado
    local_nodes_visited++;

    int N = adj.size();
    // base case is when we have reached level N which
    // means we have covered all the nodes once
    if (level==N)
    {
        // check if there is an edge from last vertex in path back to the first vertex
        if (adj[curr_path[level-1]][curr_path[0]] != 0)
        {
            // curr_res has the total weight of the solution we got
            int curr_res = curr_weight + adj[curr_path[level-1]][curr_path[0]];

            // Update final result and final path if current result is better.
            if (curr_res < final_res)
            {
                copyToFinal(curr_path);
                final_res = curr_res;
            }
        }
        return;
    }

    // for any other level iterate for all vertices to
    // build the search space tree recursively
    for (int i=0; i<N; i++)
    {
        // [PARALLELIZATION]
        // In level 1 we distribute the work.
        // Each process explores a subset of the branches starting from the root.
        // i starts at my_rank and increments by n_procs.
        if (level == 1) {
            if (i % n_procs != my_rank) continue;
        }

        // Consider next vertex if it is not same (diagonal
        // entry in adjacency matrix and not visited already)
        if (adj[curr_path[level-1]][i] != 0 &&
            visited[i] == false)
        {
            int temp = curr_bound;
            curr_weight += adj[curr_path[level-1]][i];

            // different computation of curr_bound for level 2 from the other levels
            if (level==1)
                curr_bound -= ((firstMin(adj, curr_path[level-1]) + firstMin(adj, i))/2);
            else
                curr_bound -= ((secondMin(adj, curr_path[level-1]) + firstMin(adj, i))/2);

            // curr_bound + curr_weight is the actual lower bound
            // for the node that we have arrived on
            // If current lower bound < final_res, explore further node.
            if (curr_bound + curr_weight < final_res)
            {
                curr_path[level] = i;
                visited[i] = true;

                // call TSPRec for the next level
                // Pass rank and n_procs, but they are only used at level 1
                TSPRec(adj, curr_bound, curr_weight, level+1, curr_path, my_rank, n_procs);
            }

            // Else we have to prune the node by resetting
            // all changes to curr_weight and curr_bound
            curr_weight -= adj[curr_path[level-1]][i];
            curr_bound = temp;

            // Also reset the visited array
            fill(visited.begin(), visited.end(), false);
            for (int j=0; j<=level-1; j++)
                visited[curr_path[j]] = true;
        }
    }
}
// This function sets up final_path
long long TSP(const vector<vector<int>>& adj)
{   
    int my_rank, size;
    // MPI_Init is handled in main
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    local_nodes_visited = 0;

    int N = adj.size();
    vector<int> curr_path(N + 1);
    // Calculate initial lower bound for the root node
    // using the formula 1/2 * (sum of first min + second min) for all edges.
    // firstMin is the edge with lowest weight leaving the node
    // secondMin is the second lowest weight leaving the node
    // Also initialize the curr_path and visited array
    
    int curr_bound = 0;
    
    fill(curr_path.begin(), curr_path.end(), -1);
    // Initialize globals
    visited.assign(N, false);
    final_path.assign(N + 1, 0);
    final_res = INT_MAX;
    // Compute initial bound, its like an estimation of the minimum cost
    // so we can filter some bad "branches" later
    for (int i=0; i<N; i++)
        curr_bound += (firstMin(adj, i) + secondMin(adj, i));
    
    // Rounding off the lower bound to an integer
    curr_bound = (curr_bound&1)? curr_bound/2 + 1 : curr_bound/2;

    // We start at vertex 1 so the first vertex
    // in curr_path[] is 0
    visited[0] = true;
    curr_path[0] = 0;

    // Call to TSPRec for curr_weight equal to 0 and level 1
    // Pass my_rank and size for parallelization
    TSPRec(adj, curr_bound, 0, 1, curr_path, my_rank, size);
    // After local search is done, we need to find the global minimum.
    
    struct {
        int value;
        int rank;
    } local_min, global_min;

    local_min.value = final_res;
    local_min.rank = my_rank;
    // Find the process with the minimum cost
    MPI_Allreduce(&local_min, &global_min, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

    // Update final_res to the global minimum
    final_res = global_min.value;

    // If I am the rank that found the minimum, send my path to Rank 0
    // If I am Rank 0, receive the path (if it's not me)

    if (my_rank == 0) {
        if (global_min.rank != 0) {
            // Receive path from the winner
            MPI_Recv(final_path.data(), N + 1, MPI_INT, global_min.rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // If I am the winner, final_path is already correct.
        
        printf("Minimum cost : %d\n", final_res);
        //printf("Path Taken : ");
        //for (int i=0; i<final_path.size(); i++)
        //    printf("%d ", final_path[i]);
        //printf("\n");

    } else {
        if (my_rank == global_min.rank) {
            // Send my path to Rank 0
            MPI_Send(final_path.data(), N + 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
    
    return (my_rank == 0) ? total_nodes_visited : 0;
}


int main(int argc, char** argv) {
    int my_rank;
    double t0, t1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);


    if (my_rank == 0) {
        if (argc > 1) {
            number_of_nodes = atoi(argv[1]);
        } else {
            cout << "Ingrese N : ";
            cin >> number_of_nodes;
        }
    }

    MPI_Bcast(&number_of_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);


    vector<City> cities = readTSPLIB("data/xqf131.tsp"); 

    if (cities.empty()) {
        if (my_rank == 0) cerr << "Error: No se pudo leer el archivo o esta vacio." << endl;
        MPI_Finalize();
        return 1;
    }
    
    if (number_of_nodes > cities.size()) number_of_nodes = cities.size();
    cities.resize(number_of_nodes);

    vector<vector<int>> adj = computeDistanceMatrix(cities);
    

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    
    long long total_nodes = TSP(adj); 
    
    t1 = MPI_Wtime();
    
    if (my_rank == 0) {
        double duration_sec = t1 - t0;
        double duration_ms = duration_sec * 1000.0;
        
        // Cálculo de FLOPs y GFLOPs
        double estimated_flops = (double)total_nodes * number_of_nodes * 2.0;
        double gflops_per_sec = (duration_sec > 0) ? (estimated_flops / duration_sec) / 1e9 : 0.0;

        cout << "\n=== RESULTADOS MPI  ===" << endl;
        cout << "N (Size): " << number_of_nodes << endl;
        cout << "Tiempo (ms): " << duration_ms << endl;
        cout << "Costo Minimo: " << final_res << endl;
        cout << "Nodos Visitados: " << total_nodes << endl;
        cout << "FLOPs Totales: " << (long long)estimated_flops << endl;
        cout << "GFLOPs/s: " << gflops_per_sec << endl;
        cout << "===============================\n" << endl;
    }

    MPI_Finalize();
    return 0;
}