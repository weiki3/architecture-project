#include "../include/cpu_bfs.h"

using namespace std;

void cpu_bfs(int start, Graph &my_graph, std::vector<int> &distance, std::vector<bool> &is_visited)
{
	fill(distance.begin(), distance.end(), INT_MAX);
	distance[start] = 0;
	queue<int> to_visit;
	to_visit.push(start);

	while (!to_visit.empty())
	{
		int current = to_visit.front();
		to_visit.pop();
		for (int i = my_graph.edge_offset[current]; i < my_graph.edge_offset[current] + my_graph.edges_size[current]; ++i)
		{
			int v = my_graph.adjacency_list[i];
			if (distance[v] == INT_MAX)
			{
				distance[v] = distance[current] + 1;
				to_visit.push(v);
			}
		}
	}
}
