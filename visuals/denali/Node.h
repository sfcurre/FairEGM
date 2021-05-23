#ifndef NODE_H
#define NODE_H

#include <vector>

class Edge;

class Node
{
public:
	std::vector<Node*> graph_neighbors;
	std::vector<Node*> graph_neighbors2;
	std::vector<Edge*> neighbor_edges;
	std::vector<Node*> tree_parents;
	std::vector<Node*> tree_parents2;
	int id;
	Node* cur_root;
	Node* cur_root2;
	double value;
	double value2;
	double diff_value;
	double weight;
	bool in_graph;
	bool in_graph2;
	bool marked;

	Edge* edge_min_id;
	Node* tree_node;

	int order_id;
	int set_id;

	Node() {
		weight = 1.0;
		cur_root = cur_root2 = this;
		in_graph = false;
		in_graph2 = false;
		marked = false;
		diff_value = value = value2 = 0.0;
		edge_min_id = 0;
		tree_node = 0;
	}
};

#endif
