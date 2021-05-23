#ifndef EDGE_H
#define EDGE_H

class Node;

class Edge
{
public:
	int id;
	Node* node[2];
	Node* tree_node;
	double value;
	int order_id;

	Edge(Node* n0, Node* n1) {
		node[0] = n0;
		node[1] = n1;
	}
};


#endif
