#include "Graph.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <set>

using namespace std;
int counter = 0;

void subgraphEnumeration(Graph* graph, set<int>* embedding, int k) {
	// Print the embedding
	// for(set<int>::iterator it = embedding->begin() ; it != embedding->end() ; it++) {
	// 	cout << *it << ";";
	// }
	// cout << "\n";
	if(embedding->size() == k) {
		for(set<int>::iterator it = embedding->begin() ; it != embedding->end() ; it++) {
			cout << *it << " ";
		}
		cout << "\n";
		counter++;
	}

	set<int>* extensions = new set<int>();
	for(set<int>::reverse_iterator it = embedding->rbegin() ; it != embedding->rend() ; it++) {
		int currentVertex = *it;
		set<int>* neighbours = graph->getNeighbours(currentVertex);

		for(set<int>::iterator itNeighbour = neighbours->begin() ; itNeighbour != neighbours->end() ; itNeighbour++) {
			int candidate = *itNeighbour;
			if  (
					embedding->find(candidate) == embedding->end() &&
					candidate > *(embedding->begin())
				)
			{
				if(candidate < *(embedding->rbegin())) {
					if(it == embedding->rbegin())
						extensions->insert(candidate);
					else
						extensions->erase(candidate);
				}
				else
					extensions->insert(candidate);
			}
		}
	}

	for(set<int>::iterator it = extensions->begin() ; it != extensions->end() ; it++) {
		set<int>* embeddingExtended = new set<int>(embedding->begin(), embedding->end());
		embeddingExtended->insert(*it);

		subgraphEnumeration(graph, embeddingExtended, k);

		delete embeddingExtended;
	}

	delete extensions;
}




int main(int argc, char** argv) {

	char* url = argv[1];
	int k = atoi(argv[2]);

	Graph* graph = new Graph(url);

	printf("Number of edges: %d\n", graph->getNumberOfEdges());

	set<int>* vertexes = graph->getVertexIds();
	set<int>* embedding = new set<int>();
	for(set<int>::iterator it = vertexes->begin() ; it != vertexes->end() ; it++) {
		embedding->insert(*it);

		subgraphEnumeration(graph, embedding, k);

		embedding->clear();
	}

	cout << counter << "\n";

	delete embedding;
	delete graph;

	return 0;
}
