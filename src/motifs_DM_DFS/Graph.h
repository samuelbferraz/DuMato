#ifndef GRAPH_H
#define GRAPH_H

#include <map>
#include <set>
#include <stdio.h>



class Graph {
	private:
		FILE* fp;
		std::map<int,std::set<int>*>* vertexes;
		std::set<int>* vertexIds;
		int numberOfEdges;
		int maxVertexId;
		int maxDegree;

	public:
		Graph(const char*);
		void addEdge(int src, int dst);
		std::set<int>* getNeighbours(int vertex);
		bool areNeighbours(int vertex1, int vertex2);
		int getMaxVertexId();
		int getNumberOfEdges();
		int getMaxDegree();
		std::map<int,std::set<int>*>* getVertexes();
		std::set<int>* getVertexIds();
		void print();
		~Graph();
};

#endif
