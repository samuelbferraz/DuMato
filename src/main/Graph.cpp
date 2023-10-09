#include <map>
#include <iostream>
#include <fstream>
#include "Graph.h"

Graph::Graph() {
	vertexes = new std::map<int,std::set<int>*>();
	vertexIds = new std::set<int>();

	numberOfEdges = 0;
	maxVertexId = -1;
	maxDegree = -1;

	full = false;
}

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

	edgeWeight = new std::vector<std::map<int,int>*>();
	for(int i = 0 ; i < getNumberOfVertices() ; i++)
		edgeWeight->push_back(new std::map<int,int>());

	full = true;
}

int Graph::getNumberOfVertices() {
	return maxVertexId+1;
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

void Graph::addEdgeWeight(int src, int dst, int amount) {
	// Guarantee src < dst.
	if(src > dst) {
		int aux = src;
		src = dst;
		dst = aux;
	}

	std::map<int,int>::iterator it = edgeWeight->at(src)->find(dst);
	if(it == edgeWeight->at(src)->end()) {
		edgeWeight->at(src)->insert({dst,amount});
	}
	else {
		it->second += amount; 
	}
}

int Graph::getEdgeWeight(int src, int dst) {
	// Guarantee src < dst.
	if(src > dst) {
		int aux = src;
		src = dst;
		dst = aux;
	}
	return edgeWeight->at(src)->at(dst);
}

int Graph::calculateWeight(unsigned int* subgraph, int k) {
	int sharedEdges = 0;
	for(int i = 0 ; i < k - 1 ; i++) {
		for(int j = i+1 ; j < k ; j++) { 
			int w = getEdgeWeight(subgraph[i], subgraph[j]);
			sharedEdges += (w > 1 ? 1 : 0);
		}
	}
	int weight = (((k * (k-1))/2) - sharedEdges) - (k+1); 
	return weight;
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

int Graph::getCurrentNumberOfEdges() {
	int currentNumberOfEdges = 0;
	int degree;
	for(std::map<int,std::set<int>*>::iterator it = vertexes->begin() ; it != vertexes->end() ; it++) {
		degree = it->second->size();
		currentNumberOfEdges += degree;
		if(degree > maxDegree)
			maxDegree = degree;
	}
	currentNumberOfEdges /= 2;
	return currentNumberOfEdges;
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
			if(it->first < *itSet)
				std::cout << it->first << " " << *itSet << "\n";
		}
	}
	std::cout << "\n";
}

void Graph::printToFile(const char *filename) {
	std::ofstream fp(filename);
	// std::cout << filename << "\n";
	for(std::map<int,std::set<int>*>::iterator it = vertexes->begin() ; it != vertexes->end() ; it++) {
		for(std::set<int>::iterator itSet = it->second->begin() ; itSet != it->second->end() ; itSet++) {
			if(it->first < *itSet)
				fp << (it->first)+1 << " " << (*itSet)+1 << "\n";
		}
	}
}

long unsigned int Graph::stats() {
	std::cout << "#vertices: " << vertexes->size() << "\n";
	std::cout << "#edges: " << getCurrentNumberOfEdges() << "\n";

	long unsigned int bytes = vertexes->size() * 8 + (getCurrentNumberOfEdges() * 2) * 4;
	return bytes;
}

Graph::~Graph() {
	for(std::map<int,std::set<int>*>::iterator it = vertexes->begin() ; it != vertexes->end() ; it++)
		delete it->second;
	delete vertexes;
	delete vertexIds;

	if(full) {
		fclose(fp);
		for(int i = 0 ; i < getNumberOfVertices() ; i++)
			delete edgeWeight->at(i);
		delete edgeWeight;
	}
}
