#include "../include/utils.h"

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