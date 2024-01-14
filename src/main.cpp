#include <iostream>
#include <time.h>

#include "../include/graph.h"
#include "../include/cpu_bfs.h"
#include "../include/cuda_bfs.cuh"

using namespace std;

class Checker
{
    vector<int> expected_answer;

public:
    Checker(vector<int> exp_ans) : expected_answer(exp_ans) {}

    pair<int, int> count_visited_vertices(const vector<int> &distance)
    {
        int depth = 0;
        int count = 0;
        for (int x : distance)
        {
            if (x < INT_MAX)
            {
                ++count;
                if (x > depth)
                {
                    depth = x;
                }
            }
        }
        return {count, depth};
    }

    void check(vector<int> answer)
    {
        assert(answer.size() == expected_answer.size());
        bool is_ok = true;
        int position_wrong = -1;
        for (int i = 0; i < answer.size(); ++i)
        {
            if (answer.at(i) != expected_answer.at(i))
            {
                is_ok = false;
                position_wrong = i;
                break;
            }
        }
        if (is_ok)
        {
            pair<int, int> graph_output = count_visited_vertices(answer);
            int n_visited_vertices = graph_output.first;
            int depth = graph_output.second;
            printf("CHECKED SUCCESSFULY! Number of is_visited vertices: %i, depth: %i \n", n_visited_vertices, depth);
        }
        else
        {
            printf("Something went wrong!\n");
            printf("Answer at %i equals %i but should be equal to %i\n", position_wrong, answer[position_wrong], expected_answer[position_wrong]);
        }
    }
};

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
    printf("Elapsed time for CPU implementation : %lf ms.\n", duration);

    Checker checker(distance);

    // 使用 GPU-CUDA 做 BFS
    distance = vector<int>(my_graph.vertex_num);
    start_time = clock();
    cuda_bfs(start, my_graph, distance, is_visited);
    end_time = clock();
    duration = ((double)(end_time - start_time));
    printf("Elapsed time for naive linear GPU implementation (with graph copying) : %lf ms.\n", duration);

    checker.check(distance);

    return 0;
}