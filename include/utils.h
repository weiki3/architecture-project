#ifndef UTILS_H
#define UTILS_H

#include <bits/stdc++.h>
using namespace std;

/*
 * 图类，通过临界矩阵来表示图结构
 */
class Checker
{
    vector<int> expected_answer;

public:
    Checker(vector<int> exp_ans);
    pair<int, int> get_visited_vertex(const vector<int> &distance);
    void check(vector<int> answer);
};

#endif