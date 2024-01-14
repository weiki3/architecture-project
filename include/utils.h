#ifndef UTILS_H
#define UTILS_H

#include <bits/stdc++.h>
using namespace std;

class Checker
{
    vector<int> expected_answer;

public:
    Checker(vector<int> exp_ans);
    pair<int, int> count_visited_vertices(const vector<int> &distance);
    void check(vector<int> answer);
};

#endif