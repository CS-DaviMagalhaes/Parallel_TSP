# TSP Problem

## Sequential Algorithm Explanation (Branch and Bound)

The sequential implementation of the Traveling Salesman Problem (TSP) in this repository uses the **Branch and Bound** technique to find the minimum cost tour. It does it in the following steps:

### 1. Lower Bound Calculation
The algorithm starts by calculating a "lower bound" for the starting city node. This bound is an estimate of the minimum cost to complete a tour starting from this state. We're going to explore the nodes, forming branches and then we filter (prune) the ones that are not promissing.
- It is calculated using the formula: `1/2 * sum(first_min + second_min)` for all vertices.
- `first_min`: The smallest edge weight connected to a vertex.
- `second_min`: The second smallest edge weight connected to a vertex.
- We use this to prune the searches. If a partial path's cost plus its lower bound is already greater in cost than the best solution found so far (`final_res`), that path is discarded (pruned).

### 2. Recursive Search (State Space Tree)
The algorithm explores the state space tree recursively:
- **Level 1**: Starts at the first city (index 0).
- **Subsequent Levels**: At each level, it considers moving to a next unvisited city.
- For each candidate city, it calculates:
    - `curr_weight`: The actual cost of the path so far.
    - `curr_bound`: The updated lower bound for this new node.
- **Pruning**: If `curr_bound + curr_weight < final_res`, the branch is explored further. Otherwise, it is pruned.

### 3. Updating the Solution
- When the recursion reaches the last level (all cities visited), it checks if there is a path back to the start.
- If the total cost is less than `final_res`, it updates `final_res` and stores the current path as the new best solution (`final_path`).
