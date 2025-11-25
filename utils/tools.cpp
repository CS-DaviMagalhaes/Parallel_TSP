#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
using namespace std;

struct City {
    double x, y;
};

vector<City> readTSPLIB(const string &filename) {
    ifstream file(filename);
    string line;
    vector<City> cities;
    bool readingCoords = false;

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
