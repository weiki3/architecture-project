/*
图类，通过临界矩阵来表示图结构
*/

#ifndef GRAPH_H
#define GRAPH_H

#include <bits/stdc++.h>

enum Direction
{
    Directed,
    Undirected
};
enum Format
{
    Empty,
    Edges,
    adjacency_list
};

class Graph
{
public:
    Graph();
    std::vector<int> adjacency_list; // neighbours of consecutive vertexes
    std::vector<int> edge_offset;    // offset to adjacency_list for every vertex
    std::vector<int> edges_size;     // number of edges for every vertex
    int vertex_num;
    int edge_num;

private:
    void initialize(std::vector<std::vector<int>> adjacency_list, int edge_num);
};

#endif // GRAPH_H
