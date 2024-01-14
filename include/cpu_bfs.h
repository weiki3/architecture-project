#ifndef CPU_BFS_H
#define CPU_BFS_H

#include <queue>
#include <bits/stdc++.h>

#include "./graph.h"

void cpu_bfs(int start, Graph &my_graph, std::vector<int> &distance, std::vector<bool> &is_visited);

#endif
