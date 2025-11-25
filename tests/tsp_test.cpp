#include <iostream>
#include <vector>
#include "../sequential.cpp"
#include "../utils/tools.h"
using namespace std;

int main() {
    vector<City> cities = readTSPLIB("./data/xqf131.tsp");
    vector<vector<int>> adj = computeDistanceMatrix(cities);
    int N = adj.size();

    // Llamar TSP function con matriz de adyacencia
    TSP(adj);

    printf("Minimum cost : %d\n", final_res);
    printf("Path Taken : ");
    for (int i=0; i<final_path.size(); i++)
        printf("%d ", final_path[i]);
    printf("\n");

}
