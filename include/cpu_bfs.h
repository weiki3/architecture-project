#ifndef CPU_BFS_H
#define CPU_BFS_H

#include <bits/stdc++.h>

#include "./graph.h"

// 使用 CPU 和内存完成 BFS
void cpu_bfs(int start, Graph &my_graph, std::vector<int> &distance, std::vector<bool> &is_visited);

#endif
