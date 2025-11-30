#include "../tsp_mpi.cpp"

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


    vector<City> cities = readTSPLIB("../data/xqf131.tsp"); 

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