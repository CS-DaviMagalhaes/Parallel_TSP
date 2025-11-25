#include <bits/stdc++.h>
#include "sequential.cpp"
using namespace std;

int main()
{
    //Adjacency matrix for the given graph
    vector<vector<int>> adj = { 
        {0, 10, 15, 20},
        {10, 0, 35, 25},
        {15, 35, 0, 30},
        {20, 25, 30, 0}
    };

    TSP(adj);

    printf("Minimum cost : %d\n", final_res);
    printf("Path Taken : ");
    for (int i=0; i<final_path.size(); i++)
        printf("%d ", final_path[i]);
    printf("\n");
    return 0;
}