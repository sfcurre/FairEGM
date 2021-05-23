#include "TreeBuilder.h"
#include "Node.h"

int main(int argc, char* argv[]) {	
	int i = 0;
	const char* value_file = argv[++i];
	const char* graph_file = argv[++i];
	const char* tree_file = argv[++i];
	TreeBuilder tb;	
	tb.LoadData( value_file, graph_file);
	tb.BuildScalarTree(); 
	tb.OutputTree(tree_file);


/*	int i = 0;
	const char* value_file = argv[++i];
	const char* graph_file = argv[++i];
	const char* score_file = argv[++i];
	const char* tree_file = argv[++i];
	TreeBuilder tb;	
	tb.LoadData( value_file, graph_file);
	//tb.BuildScalarTree(); 

	tb.BuildMultiCommScalarTree2(score_file);
	tb.OutputTree(tree_file); */

/*	int i = 0;
	const char* value_file = argv[++i];
	const char* graph_file = argv[++i];
	const char* new_value_file = argv[++i];
	const char* new_graph_file = argv[++i]; 
	const char* tree_file = argv[++i];
	//const char* color_file = argv[++i];

	TreeBuilder tb;	
	tb.LoadData(value_file, graph_file);
	tb.LoadDynamicDataScalar(new_value_file, new_graph_file);
	//tb.BuildMergeScalarTree();
	//tb.BuildDynamicScalarTree();
	//tb.BuildScalarTree();
	tb.BuildDiffScalarTreeG1();
	//tb.BuildDiffScalarTree_assoc();
	tb.OutputTree(tree_file); 
	//tb.OutputDiffColor(color_file);*/

	/*int i = 0;
	const char* edge_graph_file = argv[++i];
	const char* tree_file = argv[++i];
	TreeBuilder tb;	
	tb.LoadEdgeData( edge_graph_file);
	tb.BuildEdgeScalarTree2(); 
	tb.OutputTree(tree_file); */

}

