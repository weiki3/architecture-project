#include <bits/stdc++.h>
#include "../include/utils.h"
#include "../include/graph.h"
#include "../include/cpu_bfs.h"
#include "../include/cuda_bfs.cuh"

using namespace std;

int main()
{

    Graph my_graph;          // 输入节点，以邻接表的结构保存图
    vector<int> distance;    // 储存从根节点到各个节点的距离
    vector<bool> is_visited; // 储存是否遍历的信息
    int start = 1;           // 起始节点为 1

    // 使用 CPU 做 BFS
    distance = vector<int>(my_graph.vertex_num);
    is_visited = vector<bool>(my_graph.vertex_num);
    clock_t start_time = clock();
    cpu_bfs(start, my_graph, distance, is_visited);
    clock_t end_time = clock();
    double duration = ((double)(end_time - start_time));
    printf("Elapsed time for CPU implementation : %.1lf ms.\n", duration);

    // 检查结果正确性
    Checker checker(distance);

    // 使用 GPU-CUDA 做 BFS
    distance = vector<int>(my_graph.vertex_num);
    start_time = clock();
    cuda_bfs(start, my_graph, distance, is_visited);
    end_time = clock();
    duration = ((double)(end_time - start_time));
    printf("Elapsed time for naive linear GPU implementation (with graph copying) : %.1lf ms.\n", duration);

    // 检查结果正确性
    checker.check(distance);

    return 0;
}