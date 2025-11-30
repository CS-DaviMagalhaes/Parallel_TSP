#include <iostream>
#include <vector>
#include <mpi.h>
#include "../tsp_mpi.cpp"
#include "../utils/tools.cpp"
using namespace std;

int main(int argc, char** argv) {
    int my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Rank 0 reads the file and computes the distance matrix
    vector<City> cities = readTSPLIB("../data/xqf131.tsp");
    
    if (cities.empty()) {
        if (my_rank == 0) cerr << "Error reading file or file empty" << endl;
        MPI_Finalize();
        return 1;
    }
    cities.resize(20);

    vector<vector<int>> adj = computeDistanceMatrix(cities);
    TSP(adj);

    MPI_Finalize();
    return 0;
}
