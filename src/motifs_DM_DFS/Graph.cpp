#include <map>
#include <iostream>
#include "Graph.h"

Graph::Graph(const char* filename) {
	vertexes = new std::map<int,std::set<int>*>();
	vertexIds = new std::set<int>();

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
			(*vertexes)[i] = new std::set<int>();
		vertexIds->insert(i);
	}

	int degree = -1;
	for(std::map<int,std::set<int>*>::iterator it = vertexes->begin() ; it != vertexes->end() ; it++) {
		degree = it->second->size();
		numberOfEdges += degree;
		if(degree > maxDegree)
			maxDegree = degree;
	}
	numberOfEdges /= 2;
}

void Graph::addEdge(int src, int dst) {
	if(vertexes->find(src) == vertexes->end()) {
		(*vertexes)[src] = new std::set<int>();
	}
	if(vertexes->find(dst) == vertexes->end()) {
		(*vertexes)[dst] = new std::set<int>();
	}

	(*vertexes)[src]->insert(dst);
	(*vertexes)[dst]->insert(src);

	if(src > maxVertexId)
		maxVertexId = src;
	if(dst > maxVertexId)
		maxVertexId = dst;
}

std::set<int>* Graph::getNeighbours(int vertex) {
	return (*vertexes)[vertex];
}

bool Graph::areNeighbours(int vertex1, int vertex2) {
	return (*vertexes)[vertex1]->find(vertex2) != (*vertexes)[vertex1]->end();
}

int Graph::getMaxVertexId() {
	return maxVertexId;
}

int Graph::getNumberOfEdges() {
	return numberOfEdges;
}

std::map<int,std::set<int>*>* Graph::getVertexes() {
	return vertexes;
}

std::set<int>* Graph::getVertexIds() {
	return vertexIds;
}

int Graph::getMaxDegree() {
	return maxDegree;
}

void Graph::print() {
	for(std::map<int,std::set<int>*>::iterator it = vertexes->begin() ; it != vertexes->end() ; it++) {
		for(std::set<int>::iterator itSet = it->second->begin() ; itSet != it->second->end() ; itSet++) {
			std::cout << it->first << "->" << *itSet << "\n";
		}
		std::cout << "\n";
	}
}

Graph::~Graph() {
	fclose(fp);

	for(std::map<int,std::set<int>*>::iterator it = vertexes->begin() ; it != vertexes->end() ; it++)
	delete it->second;

	delete vertexes;
	delete vertexIds;
}
