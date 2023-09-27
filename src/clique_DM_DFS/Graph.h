#ifndef GRAPH_H
#define GRAPH_H

#include <map>
#include <set>
#include <stdio.h>

using namespace std;

class Graph {
	private:
		FILE* fp;
		map<int,set<int>*>* vertexes;
		set<int>* vertexIds;
		int numberOfEdges;
		int maxVertexId;
		int maxDegree;

	public:
		Graph(const char*);
		void addEdge(int src, int dst);
		set<int>* getNeighbours(int vertex);
		int getMaxVertexId();
		int getNumberOfEdges();
		int getMaxDegree();
		map<int,set<int>*>* getVertexes();
		set<int>* getVertexIds();
		void print();
		~Graph();
};

#endif
