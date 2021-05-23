#include "TreeBuilder.h"
#include <sstream>
#include <fstream>
#include <iostream>
#include<vector>
#include <math.h>
#include<algorithm>

#include <time.h>
#include <iostream>
#include <sys/time.h>

double rtclock()
{
  struct timeval Tp;
  if (gettimeofday (&Tp, NULL) == 0)  {
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
  }
  else {
    std::cout<<"Error return from gettimeofday";
  }
}

#define EPSILON 0.000001

void TreeBuilder::LoadData(const char* value_file, const char* graph_file) {
	std::string s;

	std::ifstream inputfile00 ; 
	int node_id = 0;
	inputfile00.open(value_file);
	while(std::getline(inputfile00, s)) {
		if(s.size()==0)
			break;
		else {
			std::istringstream ss(s);
			double val;
			ss>>val;
			Node* n = new Node();
			n->value = val;
			n->id = node_id;
			n->in_graph = true;
			node_id++;
			graph_nodes.push_back(n);
			tree_nodes.push_back(n);
		}
	}
	inputfile00.close();	

	std::ifstream inputfile1; 
	inputfile1.open(graph_file);
	while(std::getline(inputfile1, s)) {
		if(s.size()==0)
			break;
		else {
			std::istringstream ss(s);
			int id1, id2;
			ss>>id1;
			ss>>id2;
			Node* n1 = graph_nodes[id1];
			Node* n2 = graph_nodes[id2];
			n1->graph_neighbors.push_back(n2);
			n2->graph_neighbors.push_back(n1);
		}
	}
	inputfile1.close();

}

void TreeBuilder::LoadEdgeData(const char* graph_file) {
	std::string s;
	std::ifstream inputfile1; 
	inputfile1.open(graph_file);
	std::getline(inputfile1, s);
	std::istringstream ss0(s);
	int v, e;
	ss0>>v;
	ss0>>e;
	for(int i = 0; i < v; i++) {
		Node* n = new Node();
		n->id = i;
		graph_nodes.push_back(n);
	}
	int eid = 0;
	while(std::getline(inputfile1, s)) {
		if(s.size()==0)
			break;
		else {
			std::istringstream ss(s);
			int id1, id2;
			ss>>id1;
			ss>>id2;
			double val;
			ss>>val;
			Node* n1 = graph_nodes[id1];
			Node* n2 = graph_nodes[id2];
			Edge* e = new Edge(n1, n2);
			e->id = eid;
			e->value = val;
			e->tree_node = new Node();
			e->tree_node->id = eid;
			e->tree_node->value = val;
			tree_nodes.push_back(e->tree_node);
			eid++;
			graph_edges.push_back(e);
			n1->graph_neighbors.push_back(n2);
			n1->neighbor_edges.push_back(e);
			n2->graph_neighbors.push_back(n1);
			n2->neighbor_edges.push_back(e);
		}
	}
	inputfile1.close();
}

void TreeBuilder::LoadDynamicDataScalar( const char* new_value_file, const char* new_graph_file) {
	std::string s;
	std::ifstream inputfile01; 
	int tempid = 0;
	int node_id = graph_nodes.size();
	int old_size = graph_nodes.size();
	inputfile01.open(new_value_file);
	while(std::getline(inputfile01, s)) {
		if(s.size()==0)
			break;
		else {
			std::istringstream ss(s);
			double val;
			ss>>val;
			if (tempid < old_size) {
				graph_nodes[tempid]->value2 = val;
				graph_nodes[tempid]->in_graph2 = true;
			}
			else {
				Node* n = new Node();
				n->value2 = val;
				n->id = node_id;
				node_id++;
				n->in_graph2 = true;
				//graph_nodes.push_back(n);
				//tree_nodes.push_back(n);
			}
			tempid++;
		}
	}
	inputfile01.close();

	/*std::ifstream inputfile1; 
	inputfile1.open(new_graph_file);
	while(std::getline(inputfile1, s)) {
		if(s.size()==0)
			break;
		else {
			std::istringstream ss(s);
			int id1, id2;
			ss>>id1;
			ss>>id2;
			Node* n1 = graph_nodes[id1];
			Node* n2 = graph_nodes[id2];
			n1->graph_neighbors2.push_back(n2);
			n2->graph_neighbors2.push_back(n1);
		}
	}
	inputfile1.close();*/

}


void TreeBuilder::LoadDynamicData( const char* new_value_file, const char* new_graph_file) {
	std::string s;
	std::ifstream inputfile01; 
	int tempid = 0;
	int node_id = graph_nodes.size();
	int old_size = graph_nodes.size();
	inputfile01.open(new_value_file);
	while(std::getline(inputfile01, s)) {
		if(s.size()==0)
			break;
		else {
			std::istringstream ss(s);
			double val;
			ss>>val;
			if (tempid < old_size) {
				graph_nodes[tempid]->value2 = val;
				graph_nodes[tempid]->in_graph2 = true;
			}
			else {
				Node* n = new Node();
				n->value2 = val;
				n->id = node_id;
				node_id++;
				n->in_graph2 = true;
				graph_nodes.push_back(n);
				tree_nodes.push_back(n);
			}
			tempid++;
		}
	}
	inputfile01.close();

	std::ifstream inputfile1; 
	inputfile1.open(new_graph_file);
	while(std::getline(inputfile1, s)) {
		if(s.size()==0)
			break;
		else {
			std::istringstream ss(s);
			int id1, id2;
			ss>>id1;
			ss>>id2;
			Node* n1 = graph_nodes[id1];
			Node* n2 = graph_nodes[id2];
			n1->graph_neighbors2.push_back(n2);
			n2->graph_neighbors2.push_back(n1);
		}
	}
	inputfile1.close();

}

/*
void TreeBuilder::LoadDynamicData(const char* value_file, const char* new_value_file, const char* graph_file, const char* order_file) {
	std::string s;

	std::ifstream inputfile00 ; 
	int node_id = 0;
	inputfile00.open(value_file);
	while(std::getline(inputfile00, s)) {
		if(s.size()==0)
			break;
		else {
			std::istringstream ss(s);
			double val;
			ss>>val;
			Node* n = new Node();
			n->value = val;
			n->id = node_id;
			node_id++;
			graph_nodes.push_back(n);
			tree_nodes.push_back(n);
		}
	}
	inputfile00.close();	

	std::ifstream inputfile01; 
	int tempid = 0;
	inputfile01.open(new_value_file);
	while(std::getline(inputfile01, s)) {
		if(s.size()==0)
			break;
		else {
			std::istringstream ss(s);
			double val;
			ss>>val;
			if (tempid < graph_nodes.size()) 
				graph_nodes[tempid]->value = fabs(val - graph_nodes[tempid]->value);
			tempid++;
			
		}
	}
	inputfile01.close();

	std::ifstream inputfile1; 
	inputfile1.open(graph_file);
	while(std::getline(inputfile1, s)) {
		if(s.size()==0)
			break;
		else {
			std::istringstream ss(s);
			int id1, id2;
			ss>>id1;
			ss>>id2;
			Node* n1 = graph_nodes[id1];
			Node* n2 = graph_nodes[id2];
			n1->graph_neighbors.push_back(n2);
			n2->graph_neighbors.push_back(n1);
		}
	}
	inputfile1.close();
	
	std::ifstream inputfile2; 
	int order = 0;
	inputfile2.open(order_file);
	while(std::getline(inputfile2, s)) {
		if(s.size()==0)
			break;
		else {
			std::istringstream ss(s);
			int nid;
			ss>>nid;
			if (nid < graph_nodes.size()-1) {  //exclude null node
			//if (nid < graph_nodes.size()) {  
				graph_nodes[nid]->order_id = order;
				order++;
				ordered_nodes.push_back(graph_nodes[nid]);
			}
		}
	}
	Node* null_node = graph_nodes[graph_nodes.size()-1]; //add null node
	null_node->order_id = order;  //add null node
	null_node->value = -0.0000001; //add null node
	ordered_nodes.push_back(null_node); //add null node
	inputfile2.close();
}*/




Node* FindRoot(Node* node) {
	Node* root = node;
	while(root->cur_root != root) {
		root = root->cur_root;
	}
	Node* temp = node;
	while(temp != root) {
		Node* next = temp->cur_root;
		temp->cur_root = root;
		temp = next;
	}
	node->cur_root = root;
	return root;
}

Node* FindRoot2(Node* node) {
	Node* root = node;
	while(root->cur_root2 != root) {
		root = root->cur_root2;
	}
	Node* temp = node;
	while(temp != root) {
		Node* next = temp->cur_root2;
		temp->cur_root2 = root;
		temp = next;
	}
	return root;
}

bool compare_edge_scalar (Edge* e0, Edge* e1)
{
	return (e0->value > e1->value);
}
void TreeBuilder::BuildEdgeScalarTree() {
	double Tstartall, Tendall;
	double timecostall;	
	Tstartall = rtclock();


	std::vector<Edge*> ordered_edges = graph_edges;
	std::sort(ordered_edges.begin(), ordered_edges.end(), compare_edge_scalar);
	for(int i = 0; i < ordered_edges.size(); i++) {	
		ordered_edges[i]->order_id = i;
	}
	for(int i = 0; i < ordered_edges.size(); i++) {		
		if(i%1000000== 0)
			std::cout<<i<<std::endl;
		Edge* edge = ordered_edges[i];
		Node* node = edge->tree_node;
		std::vector<Node*> neighbor_roots;
		for(int k = 0; k < 2; k++) {
			for(int j=0; j<edge->node[k]->neighbor_edges.size(); j++) {
				Edge* neigh_edge = edge->node[k]->neighbor_edges[j];
				if (neigh_edge->order_id < edge->order_id) {
					Node* nroot = FindRoot(neigh_edge->tree_node);
					if(std::find(neighbor_roots.begin(), neighbor_roots.end(), nroot) == neighbor_roots.end())
						neighbor_roots.push_back(nroot);
				}
			}
		}
		if(neighbor_roots.size() != 0) {
			for (int k=0; k<neighbor_roots.size(); k++) {
				node->tree_parents.push_back(neighbor_roots[k]);
				neighbor_roots[k]->cur_root = node;
			}	
		}
	}

	Tendall = rtclock() ; 
	timecostall =  Tendall - Tstartall ;
	std::cout<<"Time: "<<timecostall<<std::endl;
}

void TreeBuilder::BuildEdgeScalarTree2() {
	double Tstartall, Tendall;
	double timecostall;	
	Tstartall = rtclock();

	std::vector<Edge*> ordered_edges = graph_edges;
	std::sort(ordered_edges.begin(), ordered_edges.end(), compare_edge_scalar);
	for(int i = 0; i < ordered_edges.size(); i++) {	
		ordered_edges[i]->order_id = i;
	}
	
	for(int i = 0; i < graph_nodes.size(); i++) {	
		Node* node = graph_nodes[i];
		int min_order_id = ordered_edges.size()+1;
		for(int j=0; j<node->neighbor_edges.size(); j++) {
			if(node->neighbor_edges[j]->order_id < min_order_id) {
				min_order_id = node->neighbor_edges[j]->order_id;
				node->edge_min_id =  node->neighbor_edges[j];
			}
		}	
	}	

	for(int i = 0; i < ordered_edges.size(); i++) {		
		if(i%1000000== 0)
			std::cout<<i<<std::endl;
		Edge* edge = ordered_edges[i];
		Node* node = edge->tree_node;
		std::vector<Node*> neighbor_roots;
		for(int k = 0; k < 2; k++) {
			Edge* neigh_edge = edge->node[k]->edge_min_id;
			if (neigh_edge->order_id < edge->order_id) {
				Node* nroot = FindRoot(neigh_edge->tree_node);
				if(std::find(neighbor_roots.begin(), neighbor_roots.end(), nroot) == neighbor_roots.end())
					neighbor_roots.push_back(nroot);
			}
		}
		if(neighbor_roots.size() != 0) {
			for (int k=0; k<neighbor_roots.size(); k++) {
				node->tree_parents.push_back(neighbor_roots[k]);
				neighbor_roots[k]->cur_root = node;
			}	
		}
	}

	Tendall = rtclock() ; 
	timecostall =  Tendall - Tstartall ;
	std::cout<<"Time: "<<timecostall<<std::endl;
}

bool compare_scalar (Node* n0, Node* n1)
{
	return (n0->value > n1->value);
}
void TreeBuilder::BuildScalarTree() {
	double Tstartall, Tendall;
	double timecostall;	
	Tstartall = rtclock();

	ordered_nodes.clear();
	ordered_nodes = graph_nodes;
	std::sort(ordered_nodes.begin(), ordered_nodes.end(), compare_scalar);
	for(int i = 0; i < ordered_nodes.size(); i++) {	
		ordered_nodes[i]->order_id = i;
	}
	for(int i = 0; i < ordered_nodes.size(); i++) {		
		if(i%10000 == 0)
			std::cout<<i<<std::endl;
		Node* node = ordered_nodes[i];
		std::vector<Node*> neighbor_roots;
		for(int j=0; j<node->graph_neighbors.size(); j++) {
			Node* neigh = node->graph_neighbors[j];
			if (neigh->order_id < node->order_id) {
				Node* nroot = FindRoot(neigh);
				if(std::find(neighbor_roots.begin(), neighbor_roots.end(), nroot) == neighbor_roots.end())
					neighbor_roots.push_back(nroot);
			}
		}

		if(neighbor_roots.size() != 0) {
			for (int k=0; k<neighbor_roots.size(); k++) {
				node->tree_parents.push_back(neighbor_roots[k]);
				neighbor_roots[k]->cur_root = node;
			}	
		}
	}

	Tendall = rtclock() ; 
	timecostall =  Tendall - Tstartall ;
	std::cout<<"Time: "<<timecostall<<std::endl;
}



bool CommonComm(std::vector<bool> cids1, std::vector<bool> cids2) {
	for(int i=0; i<cids1.size(); i++) {
		if(cids1[i] && cids2[i])
			return true;
	}	
	return false;
}	

void TreeBuilder::BuildMultiCommScalarTree(const char* score_file) {
	std::ifstream inputfile ; 
	inputfile.open(score_file); 
	
	if (!inputfile) 
	{
		std::cout<<"no input score";
	}
	int ori_vnum;
	inputfile>>ori_vnum;
	int vnum, cnum;
	inputfile >> vnum >> cnum;
	double threshold = 0.4;
	inputfile >> threshold;

	int vid, cid;
	double score;

	threshold = 0.4;

	std::vector<std::vector<double> > all_scores;
	std::vector<std::vector<bool> > all_cids;
	for(int i = 0; i < ori_vnum; i++) {
		std::vector<double> v1(cnum, 0.0);
		all_scores.push_back(v1);
		std::vector<bool> v2(cnum, false);
		all_cids.push_back(v2);
	}
	for(int i=0; i<vnum*cnum; i++)
	{
		inputfile >>vid >>cid>>score ;	
		all_scores[vid][cid] = score;			
	} 
	std::vector<int> max_score_id(ori_vnum, 0);
	for(int i = 0; i < ori_vnum; i++) {
		double max_val = all_scores[i][0];
		int max_id = 0;
		for(int j=0; j<cnum; j++) {
			if (all_scores[i][j] > max_val) {
				max_val = all_scores[i][j] ; 
				max_id = j;
			}
			if (all_scores[i][j] > threshold) {
				all_cids[i][j] = true;
			}
		}
		max_score_id[i] = max_id;
		all_cids[i][max_id] = true;
	}

	//std::cout<<"check1";

	ordered_nodes.clear();
	ordered_nodes = graph_nodes;
	std::sort(ordered_nodes.begin(), ordered_nodes.end(), compare_scalar);
	for(int i = 0; i < ordered_nodes.size(); i++) {	
		ordered_nodes[i]->order_id = i;
	}
	for(int i = 0; i < ordered_nodes.size(); i++) {		
		if(i%10000 == 0)
			std::cout<<i<<std::endl;
		Node* node = ordered_nodes[i];
		int max_id = max_score_id[node->id];
		std::vector<Node*> neighbor_roots;
		for(int j=0; j<node->graph_neighbors.size(); j++) {
			Node* neigh = node->graph_neighbors[j];
			//if (neigh->order_id < node->order_id) {
			//if (neigh->order_id < node->order_id && node->value < all_scores[neigh->id][max_id]) {
			//if (neigh->order_id < node->order_id && max_id == max_score_id[neigh->id] ) {
			if (neigh->order_id < node->order_id && CommonComm(all_cids[node->id], all_cids[neigh->id])) {
				Node* nroot = FindRoot(neigh);
				if(std::find(neighbor_roots.begin(), neighbor_roots.end(), nroot) == neighbor_roots.end())
					neighbor_roots.push_back(nroot);
			}
		}

		if(neighbor_roots.size() != 0) {
			for (int k=0; k<neighbor_roots.size(); k++) {
				node->tree_parents.push_back(neighbor_roots[k]);
				neighbor_roots[k]->cur_root = node;
			}	
		}
	}	
}


void TreeBuilder::BuildMultiCommScalarTree2(const char* score_file) {
	std::ifstream inputfile ; 
	inputfile.open(score_file); 
	
	if (!inputfile) 
	{
		std::cout<<"no input score";
	}
	int ori_vnum;
	inputfile>>ori_vnum;
	int vnum, cnum;
	inputfile >> vnum >> cnum;
	double threshold;
	inputfile >> threshold;

	std::ofstream output;
	output.open("./ver_color.colors");
	std::ofstream output2;
	output2.open("./tree_nodes_map.txt");


	int vid, cid;
	double score;

	threshold = -0.016863;

	std::vector<std::vector<double> > all_scores;
	std::vector<std::vector<bool> > all_cids;
	for(int i = 0; i < ori_vnum; i++) {
		std::vector<double> v1(cnum, 0.0);
		all_scores.push_back(v1);
		std::vector<bool> v2(cnum, false);
		all_cids.push_back(v2);
	}
	for(int i=0; i<vnum*cnum; i++)
	{
		inputfile >>vid >>cid>>score ;	
		all_scores[vid][cid] = score;			
	} 
	std::vector<int> max_score_id(ori_vnum, 0);
	for(int i = 0; i < ori_vnum; i++) {
		double max_val = all_scores[i][0];
		int max_id = 0;
		for(int j=0; j<cnum; j++) {
			if (all_scores[i][j] > max_val) {
				max_val = all_scores[i][j] ; 
				max_id = j;
			}
			if (all_scores[i][j] > threshold) {
				all_cids[i][j] = true;
			}
		}
		max_score_id[i] = max_id;
		all_cids[i][max_id] = true;
	}

	tree_nodes.clear();
	//std::cout<<"check1";
	for(int c = 0; c < cnum; c++) {
		for(int i = 0; i < graph_nodes.size(); i++) {	
			Node* node = graph_nodes[i];
			node->value = all_scores[i][c];
		}
		ordered_nodes.clear();
		ordered_nodes = graph_nodes;
		std::sort(ordered_nodes.begin(), ordered_nodes.end(), compare_scalar);
		for(int i = 0; i < ordered_nodes.size(); i++) {	
			ordered_nodes[i]->order_id = i;
		}
		for(int i = 0; i < ordered_nodes.size(); i++) {		
			if(i%10000 == 0)
				std::cout<<i<<std::endl;
			Node* node = ordered_nodes[i];
			int max_id = max_score_id[node->id];
			if (node->value < threshold) { // && max_id != c) {
				node->tree_node = 0;
				continue;
			}
			node->tree_node = new Node();
			node->tree_node->id = tree_nodes.size();
			tree_nodes.push_back(node->tree_node);
			node->tree_node->value = node->value;	
			output<<node->tree_node->id<<'\t'<< max_id<<std::endl;
			output2<<node->tree_node->id<<'\t'<<node->id<<std::endl;
			std::vector<Node*> neighbor_roots;
			for(int j=0; j<node->graph_neighbors.size(); j++) {
				Node* neigh = node->graph_neighbors[j];
				if (neigh->order_id < node->order_id && neigh->tree_node != 0) {
				//if (neigh->order_id < node->order_id && node->value < all_scores[neigh->id][max_id]) {
				//if (neigh->order_id < node->order_id && max_id == max_score_id[neigh->id] ) {
				//if (neigh->order_id < node->order_id && CommonComm(all_cids[node->id], all_cids[neigh->id])) {
					Node* nroot = FindRoot(neigh->tree_node);
					if(std::find(neighbor_roots.begin(), neighbor_roots.end(), nroot) == neighbor_roots.end())
						neighbor_roots.push_back(nroot);
				}
			}

			if(neighbor_roots.size() != 0) {
				for (int k=0; k<neighbor_roots.size(); k++) {
					node->tree_node->tree_parents.push_back(neighbor_roots[k]);
					neighbor_roots[k]->cur_root = node->tree_node;
				}	
			}
		}
	}	

	output.close();
	output2.close();
}


bool compare_scalar2 (Node* n0, Node* n1)
{
	return (n0->value2 > n1->value2);
}

void TreeBuilder::BuildDynamicScalarTree() {
	ordered_nodes.clear();
	ordered_nodes = graph_nodes;
	std::sort(ordered_nodes.begin(), ordered_nodes.end(), compare_scalar2);
	for(int i = 0; i < ordered_nodes.size(); i++) {	
		ordered_nodes[i]->order_id = i;
	}
	int node_id = tree_nodes.size();
	for(int i = 0; i < ordered_nodes.size(); i++) {	
		if(i%10000 == 0)
			std::cout<<i<<std::endl;
		Node* node = ordered_nodes[i];
		if (node->in_graph == false)
			continue;
		for(int j=0; j<node->graph_neighbors2.size(); j++) {
			Node* neigh = node->graph_neighbors2[j];
			if (neigh->order_id < node->order_id) {
				Node* nroot2 = FindRoot2(neigh);
				if(nroot2 != node) {
					nroot2->cur_root2 = node;
				}
			}
		}

		node->value = fabs(node->value - node->value2);
		std::vector<Node*> neighbor_roots;
		for(int j=0; j<node->graph_neighbors.size(); j++) {
			Node* neigh = node->graph_neighbors[j];
			Node* nroot2 = FindRoot2(neigh);
			if (neigh->order_id < node->order_id && nroot2 == node->cur_root2) {
				Node* nroot = FindRoot(neigh);
				if(std::find(neighbor_roots.begin(), neighbor_roots.end(), nroot) == neighbor_roots.end())
					neighbor_roots.push_back(nroot);
			}
		}

		if(neighbor_roots.size() != 0) {
			double min_neigh_value = neighbor_roots[0]->value;
			for (int k=1; k<neighbor_roots.size(); k++) {
				min_neigh_value = std::min(neighbor_roots[k]->value, min_neigh_value);
			}
			if (min_neigh_value < node->value - EPSILON) {
				Node* null_node = new Node();
				null_node->id = node_id;
				node_id++;
				tree_nodes.push_back(null_node);
				null_node->weight = 0.0;
				null_node->value = min_neigh_value;
				//update tree
				null_node->tree_parents.push_back(node);
				node->cur_root = null_node;
				for (int k=0; k<neighbor_roots.size(); k++) {
					null_node->tree_parents.push_back(neighbor_roots[k]);
					neighbor_roots[k]->cur_root = null_node;
				}
			} else {
				for (int k=0; k<neighbor_roots.size(); k++) {
					node->tree_parents.push_back(neighbor_roots[k]);
					neighbor_roots[k]->cur_root = node;
				}			
			}
		}
	}
}


void TreeBuilder::BuildMergeScalarTree() {
	ordered_nodes.clear();
	ordered_nodes = graph_nodes;
	std::sort(ordered_nodes.begin(), ordered_nodes.end(), compare_scalar2);
	for(int i = 0; i < ordered_nodes.size(); i++) {	
		ordered_nodes[i]->order_id = i;
	}
	int node_id = tree_nodes.size();
	for(int i = 0; i < ordered_nodes.size(); i++) {	
		if(i%10000 == 0)
			std::cout<<i<<std::endl;
		Node* node = ordered_nodes[i];
		if (node->in_graph == false)
			continue;
/*		for(int j=0; j<node->graph_neighbors2.size(); j++) {
			Node* neigh = node->graph_neighbors2[j];
			if (neigh->order_id < node->order_id) {
				Node* nroot2 = FindRoot2(neigh);
				if(nroot2 != node) {
					nroot2->cur_root2 = node;
				}
			}
		}*/

		//node->value = fabs(node->value - node->value2);
		node->value = node->value2;
		std::vector<Node*> neighbor_roots;
		for(int j=0; j<node->graph_neighbors.size(); j++) {
			Node* neigh = node->graph_neighbors[j];
			//Node* nroot2 = FindRoot2(neigh);
			//if (neigh->order_id < node->order_id && nroot2 == node->cur_root2) {
			if (neigh->order_id < node->order_id) {
				Node* nroot = FindRoot(neigh);
				if(std::find(neighbor_roots.begin(), neighbor_roots.end(), nroot) == neighbor_roots.end())
					neighbor_roots.push_back(nroot);
			}
		}

		if(neighbor_roots.size() != 0) {
			for (int k=0; k<neighbor_roots.size(); k++) {
				node->tree_parents.push_back(neighbor_roots[k]);
				neighbor_roots[k]->cur_root = node;
			}
			/*double min_neigh_value = neighbor_roots[0]->value;
			for (int k=1; k<neighbor_roots.size(); k++) {
				min_neigh_value = std::min(neighbor_roots[k]->value, min_neigh_value);
			}
			if (min_neigh_value < node->value - EPSILON) {
				Node* null_node = new Node();
				null_node->id = node_id;
				node_id++;
				tree_nodes.push_back(null_node);
				null_node->weight = 0.0;
				null_node->value = min_neigh_value;
				//update tree
				null_node->tree_parents.push_back(node);
				node->cur_root = null_node;
				for (int k=0; k<neighbor_roots.size(); k++) {
					null_node->tree_parents.push_back(neighbor_roots[k]);
					neighbor_roots[k]->cur_root = null_node;
				}
			} else {
				for (int k=0; k<neighbor_roots.size(); k++) {
					node->tree_parents.push_back(neighbor_roots[k]);
					neighbor_roots[k]->cur_root = node;
				}			
			}*/
		}
	}
}


void TreeBuilder::BuildDiffScalarTreeG1() {
	for(int i = 0; i < graph_nodes.size(); i++) {	
		Node* node = graph_nodes[i];
		node->value = fabs(node->value - node->value2);
	}

	ordered_nodes.clear();
	ordered_nodes = graph_nodes;
	std::sort(ordered_nodes.begin(), ordered_nodes.end(), compare_scalar);
	for(int i = 0; i < ordered_nodes.size(); i++) {	
		ordered_nodes[i]->order_id = i;
	}
	int node_id = tree_nodes.size();
	for(int i = 0; i < ordered_nodes.size(); i++) {	
		if(i%10000 == 0)
			std::cout<<i<<std::endl;
		Node* node = ordered_nodes[i];
		if (node->in_graph == false)
			continue;
/*		for(int j=0; j<node->graph_neighbors2.size(); j++) {
			Node* neigh = node->graph_neighbors2[j];
			if (neigh->order_id < node->order_id) {
				Node* nroot2 = FindRoot2(neigh);
				if(nroot2 != node) {
					nroot2->cur_root2 = node;
				}
			}
		}*/

		//node->value = fabs(node->value - node->value2);
		//node->value = node->value2;
		std::vector<Node*> neighbor_roots;
		for(int j=0; j<node->graph_neighbors.size(); j++) {
			Node* neigh = node->graph_neighbors[j];
			//Node* nroot2 = FindRoot2(neigh);
			//if (neigh->order_id < node->order_id && nroot2 == node->cur_root2) {
			if (neigh->order_id < node->order_id) {
				Node* nroot = FindRoot(neigh);
				if(std::find(neighbor_roots.begin(), neighbor_roots.end(), nroot) == neighbor_roots.end())
					neighbor_roots.push_back(nroot);
			}
		}

		if(neighbor_roots.size() != 0) {
			for (int k=0; k<neighbor_roots.size(); k++) {
				node->tree_parents.push_back(neighbor_roots[k]);
				neighbor_roots[k]->cur_root = node;
			}

		}
	}
}

void TreeBuilder::BuildDiffScalarTree() {
	for(int i = 0; i < graph_nodes.size(); i++) {	
		Node* node = graph_nodes[i];
		node->value = fabs(node->value - node->value2);
	}
	
	ordered_nodes.clear();
	ordered_nodes = graph_nodes;
	std::sort(ordered_nodes.begin(), ordered_nodes.end(), compare_scalar);
	for(int i = 0; i < ordered_nodes.size(); i++) {	
		ordered_nodes[i]->order_id = i;
	}
	for(int i = 0; i < ordered_nodes.size(); i++) {		
		if(i%10000 == 0)
			std::cout<<i<<std::endl;
		Node* node = ordered_nodes[i];
		std::vector<Node*> neighbor_roots;
		for(int j=0; j<node->graph_neighbors2.size(); j++) {
			Node* neigh = node->graph_neighbors2[j];
			if (neigh->order_id < node->order_id) {
				Node* nroot = FindRoot(neigh);
				if(std::find(neighbor_roots.begin(), neighbor_roots.end(), nroot) == neighbor_roots.end())
					neighbor_roots.push_back(nroot);
			}
		}

		if(neighbor_roots.size() != 0) {
			for (int k=0; k<neighbor_roots.size(); k++) {
				node->tree_parents.push_back(neighbor_roots[k]);
				neighbor_roots[k]->cur_root = node;
			}	
		}
	}
}

void TreeBuilder::BuildDiffScalarTree_assoc() {
	for(int i = 0; i < graph_nodes.size(); i++) {	
		Node* node = graph_nodes[i];
		node->value = fabs(node->value - node->value2);
	}

	ordered_nodes.clear();
	ordered_nodes = graph_nodes;
	std::sort(ordered_nodes.begin(), ordered_nodes.end(), compare_scalar);
	for(int i = 0; i < ordered_nodes.size(); i++) {	
		ordered_nodes[i]->order_id = i;
	}
	int node_id = tree_nodes.size();
	for(int i = 0; i < ordered_nodes.size(); i++) {	
		if(i%10000 == 0)
			std::cout<<i<<std::endl;
		Node* node = ordered_nodes[i];
		if (node->in_graph == false)
			continue;
		for(int j=0; j<node->graph_neighbors2.size(); j++) {
			Node* neigh = node->graph_neighbors2[j];
			if (neigh->order_id < node->order_id) {
				Node* nroot2 = FindRoot2(neigh);
				if(nroot2 != node) {
					nroot2->cur_root2 = node;
				}
			}
		}

		std::vector<Node*> neighbor_roots;
		for(int j=0; j<node->graph_neighbors.size(); j++) {
			Node* neigh = node->graph_neighbors[j];
			Node* nroot2 = FindRoot2(neigh);
			if (neigh->order_id < node->order_id && nroot2 == node->cur_root2) {
				Node* nroot = FindRoot(neigh);
				if(std::find(neighbor_roots.begin(), neighbor_roots.end(), nroot) == neighbor_roots.end())
					neighbor_roots.push_back(nroot);
			}
		}

		if(neighbor_roots.size() != 0) {
			for (int k=0; k<neighbor_roots.size(); k++) {
				node->tree_parents.push_back(neighbor_roots[k]);
				neighbor_roots[k]->cur_root = node;
			}			

		}
	}
}

/*
void TreeBuilder::BuildTree() {
	int node_id = tree_nodes.size();
	for(int i = 0; i < ordered_nodes.size(); i++) {	
		if(i%10000 == 0)
			std::cout<<i<<std::endl;

		Node* node = ordered_nodes[i];
		std::vector<Node*> neighbor_roots;
		for(int j=0; j<node->graph_neighbors.size(); j++) {
			Node* neigh = node->graph_neighbors[j];
			if (neigh->order_id < node->order_id) {
				Node* nroot = FindRoot(neigh);
				if(std::find(neighbor_roots.begin(), neighbor_roots.end(), nroot) == neighbor_roots.end())
					neighbor_roots.push_back(nroot);
			}
		}

		if(neighbor_roots.size() != 0) {
			double min_neigh_value = neighbor_roots[0]->value;
			for (int k=1; k<neighbor_roots.size(); k++) {
				min_neigh_value = std::min(neighbor_roots[k]->value, min_neigh_value);
			}
			if (min_neigh_value < node->value - EPSILON) {
				Node* null_node = new Node();
				null_node->id = node_id;
				node_id++;
				tree_nodes.push_back(null_node);
				null_node->weight = 0.0;
				null_node->value = min_neigh_value;
				//update tree
				null_node->tree_parents.push_back(node);
				node->cur_root = null_node;
				for (int k=0; k<neighbor_roots.size(); k++) {
					null_node->tree_parents.push_back(neighbor_roots[k]);
					neighbor_roots[k]->cur_root = null_node;
				}
			} else {
				for (int k=0; k<neighbor_roots.size(); k++) {
					node->tree_parents.push_back(neighbor_roots[k]);
					neighbor_roots[k]->cur_root = node;
				}			
			}
		}
	}
}
*/

/*void TreeBuilder::MergeTreeNodes() {
			
	
}*/

void TreeBuilder::TestTreeConnect() {
	std::vector<Node*> all_roots;
	for(int i = 0; i < tree_nodes.size(); i++) {
		Node* root = FindRoot(tree_nodes[i]);
		if (!root->marked) {
			all_roots.push_back(root);
			root->marked = true;
		}
	}

	if(all_roots.size() > 1) {
		Node* n = new Node();
		double min_val = all_roots[0]->value;
		for(int i = 0; i < all_roots.size(); i++) {
			min_val = std::min(min_val, all_roots[i]->value);	
			n->tree_parents.push_back(all_roots[i]);
		}
		n->value = min_val - 0.000001;
		n->id = tree_nodes.size();
		tree_nodes.push_back(n);
	}
}

void TreeBuilder::OutputDiffColor(const char* color_file) {
	std::ofstream output;
	output.open(color_file);
	for(int i = 0; i < graph_nodes.size(); i++) {
		if(graph_nodes[i]->in_graph2) {
			if(graph_nodes[i]->in_graph)
				output<<graph_nodes[i]->id<<'\t'<<pow((double)abs(graph_nodes[i]->value2 - graph_nodes[i]->value),0.2)<<std::endl;
			else 
				output<<graph_nodes[i]->id<<'\t'<<0 <<std::endl;
		}
	}
	output<<graph_nodes.size()<<'\t'<<-0.01 <<std::endl;
	output.close();
}

void TreeBuilder::OutputTree(const char* tree_file) {
	TestTreeConnect() ;	
	
	std::ofstream output;
	output.open(tree_file);
	output<<tree_nodes.size()<<std::endl;
	for(int i = 0; i < tree_nodes.size(); i++) {
		output<<tree_nodes[i]->id<<'\t'<<tree_nodes[i]->value<<std::endl;
	}
	for(int i = 0; i < tree_nodes.size(); i++) {
		Node* tnode = tree_nodes[i];
		for (int j = 0; j < tnode->tree_parents.size(); j++) {
			output<<tnode->id<<'\t'<<tnode->tree_parents[j]->id<<std::endl;
		}
	}
	output.close();
}

