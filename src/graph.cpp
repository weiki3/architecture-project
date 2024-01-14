#include "../include/graph.h"

using namespace std;

// Graph 类的默认构造函数
Graph::Graph()
{
	int vertex_num, edge_num;
	string line;

	// 输出提示信息，开始读取输入图结构
	cout << "开始读取输入图结构" << endl;

	// 从标准输入读取顶点数量和边数量
	cin >> vertex_num >> edge_num;

	// 创建邻接表
	vector<vector<int>> adjacency_list(vertex_num);

	// 逐行读取输入，构建邻接表
	for (int v = 0; v < vertex_num; ++v)
	{
		getline(cin, line);
		stringstream splitter(line);
		int w;

		// 将每一行的数字添加到对应的邻接表中
		while (splitter >> w)
			adjacency_list[v].push_back(w);
	}

	// 调用初始化函数
	this->initialize(adjacency_list, edge_num);

	// 输出读取完毕信息
	cout << "读取完毕" << endl;
}

// 初始化图的函数
void Graph::initialize(vector<vector<int>> adjacency_list, int edge_num)
{
	const int vertex_num = adjacency_list.size();

	// 遍历邻接表，构建图的数据结构
	for (int i = 0; i < vertex_num; ++i)
	{
		this->edge_offset.push_back(this->adjacency_list.size());
		this->edges_size.push_back(adjacency_list[i].size());

		// 将邻接表中的顶点添加到图的边列表中
		for (int v : adjacency_list[i])
			this->adjacency_list.push_back(v);
	}

	// 设置图的顶点数量和边数量
	this->vertex_num = vertex_num;
	this->edge_num = edge_num;
}
