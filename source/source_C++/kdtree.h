#ifndef KDTREE_H
#define KDTREE_H

#include <iostream>
#include <vector>
#include <utility>

using namespace std;

void print_slice(vector<vector<double> > &out);
void print_point(vector<double> &point);
vector<vector<double> > get_slice(const vector<vector<double> > &data, int start, int end);
vector<double> elementwise_extreme(const vector<double> &a, const vector<double> &b, bool max);
vector<double> get_closest_point(const vector<double> &lower_bounding_planes, const vector<double> &upper_bounding_planes, const vector<double> &query, int dim);
vector<double> get_furthest_point(const vector<double> &lower_bounding_planes, const vector<double> &upper_bounding_planes, const vector<double> &query, int dim);

class KDTree {
    public:
        struct point_comparator
        {
            int discriminator;
            explicit point_comparator(int a) : discriminator(a) { }
            bool operator()( const std::vector<double>& p1, const std::vector<double>& p2 ) {
    	        return p1[discriminator] < p2[discriminator];
            }
        };                
        struct Node {
            int axis;
            int depth;
            Node *left;
            Node *right;
            vector<double> data_point;
            vector<double> mean_subtree_point;
            vector<double> lower_bounding_planes;
            vector<double> upper_bounding_planes;
            double box_length;
            double total_weight;
            double weight;
            int subtree_node_count;
            Node() {
                this->left = NULL;
                this->right = NULL;
                this->subtree_node_count = 0;
                this->depth = -1;
            }
            double KDE_point_eval(vector<double> query_point, double bw, double delta_exclude, double delta_mean, int dim);
            pair<double, double> get_approximation_costs(vector<double> query_point, double bw, int dim);
            void print();
        };    
        struct NodeCost {
            Node *node;
            double kmin_approx_cost;
            double kmax_approx_cost;
            double max_approx_cost;
            double dist;
            NodeCost() {}
            NodeCost(Node& n, double kmin_approx_cost, double kmax_approx_cost, double dist = 0.0) {
                this->kmin_approx_cost = kmin_approx_cost;
                this->kmax_approx_cost = kmax_approx_cost;
                this->max_approx_cost = max(kmin_approx_cost, kmax_approx_cost);
                this->node = &n;
                this->dist = dist;
            }
            void reset(Node& n, double kmin_approx_cost, double kmax_approx_cost, double dist = 0.0) {
                this->kmin_approx_cost = kmin_approx_cost;
                this->kmax_approx_cost = kmax_approx_cost;
                this->max_approx_cost = max(kmin_approx_cost, kmax_approx_cost);
                this->node = &n;
                this->dist = dist;
            }
        };
        struct priority_comparator {
            const bool by_dist;            
            priority_comparator(const bool by_distance) : by_dist(by_distance) {}
            bool operator()( const NodeCost& n1, const NodeCost& n2 ) {
                if (by_dist) {
                    return n1.dist > n2.dist;
                } else {
                    // I realized that the delta_rem is going to be changing as we pop things off the queue
                    // and it actually doesn't come into the comparison calculation, since all that differs 
                    // among nodes is delta_inc (cost) and number of subtree nodes. We will check our hard 
                    // constraint holds when we pop things off the queue.
                    return n1.max_approx_cost / n1.node->subtree_node_count > n2.max_approx_cost / n2.node->subtree_node_count;
                }
            }
        };       
        struct priority_comparator_d
        {
            bool operator()( const NodeCost& n1, const NodeCost& n2 ) {             
    	        return n1.dist > n2.dist;
            }
        }; 
    private:
        Node* root;
        int dim;
        bool weighted;
        vector<double> weights;
        void KDtree();
        vector<vector<double> > data_points;
        Node* build_recursive(int first, int last, int depth);        
        pair<double, pair<double, double> > pruned_kde_point_eval(vector<double> query_point, Node *subtree, double bw, double delta, bool by_dist);
    public:
        void print_tree();
        vector<vector<double> > convert_data_points(vector<double> data, int dim, bool row_major);
        void build(vector<double> data, int num_of_dimensions, bool row_major);
        void build(vector<double> data, int num_of_dimensions, bool row_major, vector<double> weights);
        vector<double> KDE_eval(const vector<vector<double> > &query_points, double bw, double delta_exclude, double delta_mean);
        vector<pair<double, pair<double, double> > > KDE_eval(const vector<vector<double> > &query_points, double bw, double delta, bool dist_based);        
};

#endif
