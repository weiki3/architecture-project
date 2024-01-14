#include "../include/utils.h"

Checker::Checker(vector<int> expected_answer) : expected_answer(expected_answer) {}

// 获取访问过的顶点数量和深度的辅助函数
pair<int, int> Checker::get_visited_vertex(const vector<int> &distance)
{
    int depth = 0;
    int vertex_num = 0;

    // 遍历距离数组，计算访问过的顶点数量和最大深度
    for (int i : distance)
    {
        if (i < INT_MAX)
        {
            ++vertex_num;
            if (i > depth)
                depth = i;
        }
    }
    // 返回结果
    return {vertex_num, depth};
}

// 检查算法输出结果的主要函数
void Checker::check(vector<int> answer)
{
    bool pass = true;
    int position_wrong = -1;

    // 遍历答案和期望答案的数组，检查是否相符
    for (int i = 0; i < answer.size(); ++i)
    {
        if (answer.at(i) != expected_answer.at(i))
        {
            pass = false;
            position_wrong = i;
            break;
        }
    }

    // 根据检查结果输出信息
    if (pass)
    {
        // 获取访问节点信息
        pair<int, int> graph_output = get_visited_vertex(answer);
        int n_visited_vertices = graph_output.first;
        int depth = graph_output.second;

        printf("结果通过检查! 图节点数量: %d, 图深度: %d \n", n_visited_vertices, depth);
    }
    else
        printf("结果错误\n");
}
