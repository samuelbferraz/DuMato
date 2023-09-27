#include "Graph.h"
#include <map>
#include <iostream>
using namespace std;

Graph::Graph(const char* filename) {
	vertexes = new map<int,set<int>*>();
	vertexIds = new set<int>();

	numberOfEdges = 0;
	maxVertexId = -1;
	maxDegree = -1;

	fp = fopen(filename, "r");

	// Reading graph
	int src, dst;
	while(fscanf(fp, "%d %d", &src, &dst) != EOF) {
		//printf("%d->%d\n", src, dst);
		this->addEdge(src, dst);
	}

	for(int i = 0 ; i <= getMaxVertexId() ; i++) {
		if(vertexes->find(i) == vertexes->end())
			(*vertexes)[i] = new set<int>();
		vertexIds->insert(i);
	}

	int degree = -1;
	for(map<int,set<int>*>::iterator it = vertexes->begin() ; it != vertexes->end() ; it++) {
		degree = it->second->size();
		numberOfEdges += degree;
		if(degree > maxDegree)
			maxDegree = degree;
	}
	numberOfEdges /= 2;
}

void Graph::addEdge(int src, int dst) {
	if(vertexes->find(src) == vertexes->end()) {
		(*vertexes)[src] = new set<int>();
	}
	if(vertexes->find(dst) == vertexes->end()) {
		(*vertexes)[dst] = new set<int>();
	}

	(*vertexes)[src]->insert(dst);
	(*vertexes)[dst]->insert(src);

	if(src > maxVertexId)
		maxVertexId = src;
	if(dst > maxVertexId)
		maxVertexId = dst;
}

set<int>* Graph::getNeighbours(int vertex) {
	return (*vertexes)[vertex];
}

int Graph::getMaxVertexId() {
	return maxVertexId;
}

int Graph::getNumberOfEdges() {
	return numberOfEdges;
}

map<int,set<int>*>* Graph::getVertexes() {
	return vertexes;
}

set<int>* Graph::getVertexIds() {
	return vertexIds;
}

int Graph::getMaxDegree() {
	return maxDegree;
}

void Graph::print() {
	for(map<int,set<int>*>::iterator it = vertexes->begin() ; it != vertexes->end() ; it++) {
		for(set<int>::iterator itSet = it->second->begin() ; itSet != it->second->end() ; itSet++) {
			cout << it->first << "->" << *itSet << "\n";
		}
		cout << "\n";
	}
}

Graph::~Graph() {
	fclose(fp);

	for(map<int,set<int>*>::iterator it = vertexes->begin() ; it != vertexes->end() ; it++)
	delete it->second;

	delete vertexes;
	delete vertexIds;
}
