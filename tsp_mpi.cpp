#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

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
        if (i == j)
            continue;

        if (adj[i][j] <= first)
        {
            second = first;
            first = adj[i][j];
        }
        else if (adj[i][j] <= second &&
                 adj[i][j] != first)
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
void TSP(const vector<vector<int>>& adj)
{   
    int my_rank, size;
    // MPI_Init is handled in main
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

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
        printf("Path Taken : ");
        for (int i=0; i<final_path.size(); i++)
            printf("%d ", final_path[i]);
        printf("\n");

    } else {
        if (my_rank == global_min.rank) {
            // Send my path to Rank 0
            MPI_Send(final_path.data(), N + 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
}