#ifndef TREEBUILDER_H
#define TREEBUILDER_H

#include "Node.h"
#include "Edge.h"

#include<vector>

class TreeBuilder
{
public:
	std::vector<Node*> graph_nodes;
	std::vector<Edge*> graph_edges;
	std::vector<Node*> ordered_nodes;
	std::vector<Node*> tree_nodes;
	std::vector<std::vector<Node*> > nodes_set;
	std::vector<Node*> set_min_nodes;

	std::vector<Node*> super_tree_nodes;

	void LoadData(const char* value_file, const char* graph_file);
	void LoadEdgeData(const char* graph_file);
	void LoadDynamicData(const char* new_value_file, const char* new_graph_file);
	void LoadDynamicDataScalar( const char* new_value_file, const char* new_graph_file) ;
	void BuildTree();
	int MergeSets(std::vector<int> set_ids);
	void OutputTree(const char* tree_file);

	void TestTreeConnect();
	void BuildScalarTree();
	void BuildEdgeScalarTree();
	void BuildDynamicScalarTree();
	void BuildMergeScalarTree();
	void BuildMultiCommScalarTree(const char* score_file) ;

	void BuildDiffScalarTree();
	void BuildDiffScalarTreeG1();
	void BuildDiffScalarTree_assoc(); 
	void BuildEdgeScalarTree2(); 
	void BuildMultiCommScalarTree2(const char* score_file) ;
	
	void OutputDiffColor(const char* color_file);
};

#endif
