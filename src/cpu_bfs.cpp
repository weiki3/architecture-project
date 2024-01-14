#include "../include/cpu_bfs.h"

using namespace std;

// 使用广度优先搜索算法在 CPU 上遍历图
void cpu_bfs(int start, Graph &my_graph, std::vector<int> &distance, std::vector<bool> &is_visited)
{
	// 初始化距离数组，将起始点距离设为0
	fill(distance.begin(), distance.end(), INT_MAX);
	distance[start] = 0;

	// 使用队列进行广度优先搜索
	queue<int> to_visit;
	to_visit.push(start);

	while (!to_visit.empty())
	{
		// 获取当前访问的节点
		int current = to_visit.front();
		to_visit.pop();

		// 遍历当前节点的邻接节点
		for (int i = my_graph.edge_offset[current]; i < my_graph.edge_offset[current] + my_graph.edges_size[current]; ++i)
		{
			int v = my_graph.adjacency_list[i];
			// 如果邻接节点的距离为无穷大，更新距离并将其加入队列
			if (distance[v] == INT_MAX)
			{
				distance[v] = distance[current] + 1;
				to_visit.push(v);
			}
		}
	}
}
