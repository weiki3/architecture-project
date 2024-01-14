#ifndef GRAPH_H
#define GRAPH_H

#include <bits/stdc++.h>

/*
 * 图类，通过临界矩阵来表示图结构
 */
class Graph
{
public:
    Graph();
    std::vector<int> adjacency_list; // 邻接节点
    std::vector<int> edge_offset;    // 每个节点在邻接节点数组中的偏移量
    std::vector<int> edges_size;     // 每个所连的边的个数
    int vertex_num;
    int edge_num;

private:
    void initialize(std::vector<std::vector<int>> adjacency_list, int edge_num);
};

#endif
