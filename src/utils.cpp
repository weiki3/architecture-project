#include "../include/utils.h"

Checker::Checker(vector<int> exp_ans) : expected_answer(exp_ans) {}

pair<int, int> Checker::get_visited_vertex(const vector<int> &distance)
{
    int depth = 0;
    int vertex_num = 0;
    for (int x : distance)
    {
        if (x < INT_MAX)
        {
            ++vertex_num;
            if (x > depth)
                depth = x;
        }
    }
    return {vertex_num, depth};
}

void Checker::check(vector<int> answer)
{
    assert(answer.size() == expected_answer.size());
    bool pass = true;
    int position_wrong = -1;
    for (int i = 0; i < answer.size(); ++i)
    {
        if (answer.at(i) != expected_answer.at(i))
        {
            pass = false;
            position_wrong = i;
            break;
        }
    }
    if (pass)
    {
        pair<int, int> graph_output = get_visited_vertex(answer);
        int n_visited_vertices = graph_output.first;
        int depth = graph_output.second;
        printf("结果通过检查! 图节点数量: %d, 图深度: %d \n", n_visited_vertices, depth);
    }
    else
    {
        printf("结果错误\n");
    }
}