#include "../include/graph.h"

using namespace std;

Graph::Graph()
{
	int vertex_num, edge_num;
	string line;

	cout << "开始读取输入图结构" << endl;
	cin >> vertex_num >> edge_num;
	vector<vector<int>> adjacency_list(vertex_num);

	for (int v = 0; v < vertex_num; ++v)
	{
		getline(cin, line);
		stringstream splitter(line);
		int w;
		while (splitter >> w)
			adjacency_list[v].push_back(w);
	}
	this->initialize(adjacency_list, edge_num);
	cout << "读取完毕" << endl;
}

void Graph::initialize(vector<vector<int>> adjacency_list, int edge_num)
{
	const int vertex_num = adjacency_list.size();
	for (int i = 0; i < vertex_num; ++i)
	{
		this->edge_offset.push_back(this->adjacency_list.size());
		this->edges_size.push_back(adjacency_list[i].size());
		for (int v : adjacency_list[i])
			this->adjacency_list.push_back(v);
	}
	this->vertex_num = vertex_num;
	this->edge_num = edge_num;
}