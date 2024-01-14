#ifndef BFS_GPU_CUH
#define BFS_GPU_CUH

#include <bits/stdc++.h>

#include "./graph.h"

void bfsGPU(int start, Graph &G, std::vector<int> &distance, std::vector<bool> &visited);

#endif // BFS_GPU_CUH
