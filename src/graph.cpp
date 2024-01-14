#include "../include/graph.h"

using namespace std;

Graph::Graph()
{
	int numVertices, numEdges;
	cout << "Started reading graph" << endl;
	cin >> numVertices >> numEdges;
	vector<vector<int>> adjacencyList(numVertices);
	string line;
	for (int v = 0; v < numVertices; ++v)
	{
		getline(cin, line);
		stringstream splitter(line);
		int w;
		while (splitter >> w)
			adjacencyList[v].push_back(w);
	}
	this->init(adjacencyList, numEdges);
	cout << "Finished reading graph" << endl;
}

void Graph::init(vector<vector<int>> adjacencyList, int numEdges)
{
	const int numVertices = adjacencyList.size();
	// Creation of single vector adjacency list
	for (int i = 0; i < numVertices; ++i)
	{
		this->edgesOffset.push_back(this->adjacencyList.size());
		this->edgesSize.push_back(adjacencyList[i].size());
		for (int v : adjacencyList[i])
		{
			this->adjacencyList.push_back(v);
		}
	}
	this->numVertices = numVertices;
	this->numEdges = numEdges;
}