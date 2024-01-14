#ifndef CUDA_BFS_H
#define CUDA_BFS_H

#include "./graph.h"
#include <bits/stdc++.h>


void cuda_bfs(int start, Graph &my_graph, std::vector<int> &distance, std::vector<bool> &is_visited);

#endif
