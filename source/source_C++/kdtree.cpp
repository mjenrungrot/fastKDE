#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <queue>
#include <cmath>
#include <assert.h>
#include <utility>
#include "kdtree.h"
#include "kernel_help.h"

#define PI 3.14159265359

// let's print internal state stuff only if debugging
#ifdef DEBUG
#define DEBUG_CODE(x) do{ x } while( false )
#else
#define DEBUG_CODE(x) do{ } while ( false )
#endif

using namespace std;

/*********************
 * Utility Functions
 *********************/


template <typename T>
void operator+=(vector<T>& a, const vector<T>& b) {
    assert(a.size() == b.size());    
    for (int i = 0; i < a.size(); i++) {
        a[i] = a[i] + b[i];
    }
}

template <typename T>
void operator/=(vector<T>& a, const int& b) { 
    for (int i = 0; i < a.size(); i++) {
        a[i] = a[i] / b;
    }
}

template <typename T>
void operator*=(vector<T>& a, const int& b) {    
    for (int i = 0; i < a.size(); i++) {
        a[i] = a[i] * b;
    }
}



void print_slice(vector<vector<double> > &out) {
    for (int i = 0; i < out.size(); i++) {
        for (int j = 0; j < out[i].size(); j++) {
            cout << out[i][j] << "\t";
        }
        cout << endl;
    }
    cout << "*******************************************" <<endl;
}

void print_point(vector<double> &point) {
    for (int i = 0; i < point.size(); i++) {
        cout << point[i] << "\t";
    }
    cout << endl;
    cout << "*******************************************" << endl;
}

vector<vector<double> > get_slice(const vector<vector<double> > &data, int start, int end) {
    vector<vector<double> > slice(&data[start], &data[end]);
    return slice;
}

vector<double> elementwise_extreme(const vector<double> &a, const vector<double> &b, bool max = true) {
    vector<double> out(a.size());
    assert(a.size() == b.size());
    for (int i = 0; i < a.size(); i++) {
        if (max) {
            out[i] = a[i] > b[i] ? a[i] : b[i];            
        } else {
            out[i] = a[i] < b[i] ? a[i] : b[i];
        }
    }
    return out;
}

vector<double> get_closest_point(const vector<double> &lower_bounding_planes, const vector<double> &upper_bounding_planes, const vector<double> &query, int dim){
    vector<double> close(lower_bounding_planes.size());
    for(int i=0; i < dim; i++) {
        close[i] = max(min(upper_bounding_planes[i],query[i]),lower_bounding_planes[i]);
    }
    return close;
}

vector<double> get_furthest_point(const vector<double> &lower_bounding_planes, const vector<double> &upper_bounding_planes, const vector<double> &query, int dim){
    vector<double> close(lower_bounding_planes.size());
    for(int i = 0; i < dim; i++) {
        close[i] = abs(upper_bounding_planes[i]-query[i]) > abs(lower_bounding_planes[i]-query[i]) ? upper_bounding_planes[i] : lower_bounding_planes[i];
    }
    return close;
}

double L2_distance(const vector<double> &a, const vector<double> &b) {
    double out = 0.0;
    assert(a.size() == b.size()); 
    for (int i = 0; i < a.size(); i++) {
        out += pow(a[i] - b[i], 2);
    }
    return sqrt(out);
}

/**********************
 * Node Methods
 **********************/

void KDTree::Node::print() {
        cout << "==============================================" << endl;
        cout << "Node: "; print_point(this->data_point);
        cout << "Depth " << this->depth << endl;
        cout << "Weight " << this->weight << endl;
        cout << "Total Weight " << this->total_weight << endl;
        cout << "Subtree Node Count " << this->subtree_node_count << endl;
        cout << "Mean  point  "; print_point(this->mean_subtree_point);
        cout << "Lower planes "; print_point(this->lower_bounding_planes);
        cout << "Upper planes "; print_point(this->upper_bounding_planes);
        cout << "==============================================" << endl;
}

pair<double, double> KDTree::Node::get_approximation_costs(vector<double> query_point, double bw, int dim) { 
    vector<double> closest = get_closest_point(this->lower_bounding_planes, this->upper_bounding_planes, query_point, dim);
    vector<double> furthest = get_furthest_point(this->lower_bounding_planes, this->upper_bounding_planes, query_point, dim);
    double max_wkde = this->total_weight * gauss_kern(query_point.begin(), closest.begin(), bw, dim);
    double min_wkde = this->total_weight * gauss_kern(query_point.begin(), furthest.begin(), bw, dim);
    double mean_wkde = this->total_weight * gauss_kern(query_point.begin(), this->mean_subtree_point.begin(), bw, dim);
    return make_pair(mean_wkde - min_wkde, max_wkde - mean_wkde);
}

/*********************
 * KDTree Methods
 *********************/

void KDTree::KDtree() {
    this->dim = 0;
    this->weighted = false;
}

vector<vector<double> > KDTree::convert_data_points(vector<double> data, int dim, bool row_major = true) {
    // Takes linear vector representation and converts to two-dimensional vector
    int num_rows = data.size() / dim;
    vector<vector<double> > out(num_rows, vector<double>(dim));
    if (row_major) {
        //Row-major order
        for (int i = 0; i < num_rows; i++) {
            for (int j = 0; j < dim; j++) {
                out[i][j] = data[i * dim + j];
            }
        }
    } else {
        //Column-major order
        for (int i = 0; i < num_rows; i++) {
            for (int j = 0; j < dim; j++) {
                out[i][j] = data[i + num_rows * j];
            }
        }
    }
    return out;
}

KDTree::Node* KDTree::build_recursive(int first, int last, int depth = 0) {
    int size = last - first + 1;
    int median = size / 2;
    int median_offset = first + median;
    int axis = depth % this->dim;
    vector<double> left_cum_sum(this->dim, 0.0);
    vector<double> right_cum_sum(this->dim, 0.0);

    Node* out = new Node();
    out->axis = axis;
    out->depth = depth;
    out->subtree_node_count = 1;
    out->mean_subtree_point.resize(this->dim, 0.0);    
    
    if (!this->weighted || this->weights.size() == 0) {    
        out->weight = 1.0/((double)this->data_points.size());
    } else {
        out->weight = this->weights[median_offset];
    }

    out->total_weight = out->weight;

    if (first == last) {
        out->data_point = this->data_points[first];
        out->mean_subtree_point = this->data_points[first];
        out->lower_bounding_planes = this->data_points[first];
        out->upper_bounding_planes = this->data_points[first];
        out->box_length = L2_distance(out->lower_bounding_planes, 
                                      out->upper_bounding_planes);
        return out;
    }    

    if (last - first > 1) { 
        sort(&this->data_points[first], &this->data_points[last], KDTree::point_comparator(axis));
    }       

    if (median_offset - 1 >= first) {
        out->left = this->build_recursive(first, median_offset - 1, depth + 1);
        left_cum_sum += out->left->mean_subtree_point;
        left_cum_sum *= out->left->subtree_node_count;
        out->mean_subtree_point += left_cum_sum;
        out->subtree_node_count += out->left->subtree_node_count;
        out->total_weight += out->left->total_weight;
    }
    if (median_offset + 1 <= last) {
        out->right = this->build_recursive(median_offset + 1, last, depth + 1);
        right_cum_sum += out->right->mean_subtree_point;
        right_cum_sum *= out->right->subtree_node_count;
        out->mean_subtree_point += right_cum_sum;
        out->subtree_node_count += out->right->subtree_node_count;
        out->total_weight += out->right->total_weight;
    }

    out->data_point = this->data_points[median_offset]; 
    out->mean_subtree_point += out->data_point;         
    out->mean_subtree_point /= out->subtree_node_count;  
    out->lower_bounding_planes = out->data_point;
    out->upper_bounding_planes = out->data_point;
    if (out->left != NULL) {
        out->lower_bounding_planes = elementwise_extreme(out->lower_bounding_planes,
                                                         out->left->lower_bounding_planes,
                                                         false);
        out->upper_bounding_planes = elementwise_extreme(out->upper_bounding_planes,
                                                         out->left->upper_bounding_planes,
                                                         true);
    }
    if (out->right != NULL) {
        out->lower_bounding_planes = elementwise_extreme(out->lower_bounding_planes,
                                                         out->right->lower_bounding_planes,
                                                         false);
        out->upper_bounding_planes = elementwise_extreme(out->upper_bounding_planes,
                                                         out->right->upper_bounding_planes,
                                                         true);
    }
    out->box_length = L2_distance(out->lower_bounding_planes, 
                                  out->upper_bounding_planes);
    return out;
}

void KDTree::print_tree() {
    queue<Node> to_visit;
    to_visit.push(*(this->root));
    Node current;

    while (!to_visit.empty()) {
        current = to_visit.front();
        to_visit.pop();
        cout << "Box Length " << current.box_length << endl;
        cout << "Depth " << current.depth << " Node: "; print_point(current.data_point);
        cout << "Mean  point  "; print_point(current.mean_subtree_point);
        cout << "Lower planes "; print_point(current.lower_bounding_planes);
        cout << "Upper planes "; print_point(current.upper_bounding_planes);        
        if (current.left != NULL) {
            to_visit.push(*current.left);
        }
        if (current.right != NULL) {
            to_visit.push(*current.right);
        }
    }
}

void KDTree::build(vector<double> data, int num_of_dimensions, bool row_major) {
    this->data_points = this->convert_data_points(data, num_of_dimensions, row_major);
    this->dim = num_of_dimensions;
    this->root = this->build_recursive(0, this->data_points.size() - 1, 0);
}

void KDTree::build(vector<double> data, int num_of_dimensions, bool row_major, vector<double> weights) {
    this->weighted = true;
    this->weights.insert(this->weights.begin(),weights.begin(),weights.end());
    this->build(data, num_of_dimensions, row_major);
}

double KDTree::Node::KDE_point_eval(vector<double> query_point, double bw, double delta_exclude, double delta_mean, int dim){
    double kdev = 0;
    double max_kde = 0;
    double min_kde = 0;
    //cout << ">> " << this->data_point[0] << " " << this->data_point[1] << " " << this->data_point[2] << "\n"; 
    //cout << "\t " << (this->weight) << "*"<< gauss_kern(query_point.begin(),this->data_point.begin(),bw,dim)<< "="<< (this->weight)*gauss_kern(query_point.begin(),this->data_point.begin(),bw,dim) << "\n";
    
    max_kde = gauss_kern(query_point.begin(),get_closest_point(this->lower_bounding_planes, this->upper_bounding_planes, query_point, dim).begin(),bw,dim);
    min_kde = gauss_kern(query_point.begin(),get_furthest_point(this->lower_bounding_planes, this->upper_bounding_planes, query_point, dim).begin(),bw,dim);
    //cout << "\t max_kde: " << max_kde << "\tmin_kde: "<< min_kde << "\tdiff: " << max_kde-min_kde << " \n";
    if((this->left==NULL && this->right==NULL) || max_kde-min_kde<=delta_mean){
        //cout << "\t MEAN " << this->data_point[0] << " " << this->data_point[1] << " " << this->data_point[2] << "\n"; 
        return (this->total_weight)*gauss_kern(query_point.begin(),this->data_point.begin(),bw,dim);
    }
    
    if(this->left!=NULL){
        //cout << "\t left max:" << (this->left->total_weight)*gauss_kern(query_point.begin(),get_closest_point(this->left->lower_bounding_planes, this->left->upper_bounding_planes, query_point, dim).begin(),bw,dim) << "\n";
    }
    
    if(this->left!=NULL && (this->left->total_weight)*gauss_kern(query_point.begin(),get_closest_point(this->left->lower_bounding_planes, this->left->upper_bounding_planes, query_point, dim).begin(),bw,dim)>delta_exclude ){
        kdev += this->left->KDE_point_eval(query_point,bw,delta_exclude,delta_mean,dim);
    } else {
        //cout << "\t LEXC " << this->data_point[0] << " " << this->data_point[1] << " " << this->data_point[2] << "\n"; 
    }
    
    if(this->right!=NULL){
        //cout << "\t right max:" << (this->right->total_weight)*gauss_kern(query_point.begin(),get_closest_point(this->right->lower_bounding_planes, this->right->upper_bounding_planes, query_point, dim).begin(),bw,dim) << "\n";
    }
    
    if(this->right!=NULL && (this->right->total_weight)*gauss_kern(query_point.begin(),get_closest_point(this->right->lower_bounding_planes, this->right->upper_bounding_planes, query_point, dim).begin(),bw,dim)>delta_exclude ){
        kdev += this->right->KDE_point_eval(query_point,bw,delta_exclude,delta_mean,dim);
    } else {
        //cout << "\t REXC " << this->data_point[0] << " " << this->data_point[1] << " " << this->data_point[2] << "\n"; 
    }
    
    kdev += (this->weight) * gauss_kern(query_point.begin(),this->data_point.begin(),bw,dim);
    return kdev;
}

pair<double, pair<double, double> > KDTree::pruned_kde_point_eval(vector<double> query_point, KDTree::Node *subtree, double bw, double delta, bool by_dist = false) {
    double delta_remaining = delta;
    double accumulated_lower_error_dist = 0.0;
    double accumulated_upper_error_dist = 0.0;
    double wkde_total = 0.0;    
    double contribution = 0.0;
    double dist = 0.0;
    KDTree::NodeCost current;   
    pair<double, double> approx_costs = subtree->get_approximation_costs(query_point, bw, this->dim);    
    priority_queue<KDTree::NodeCost, vector<KDTree::NodeCost>, KDTree::priority_comparator> nodes_to_visit(by_dist);
    KDTree::NodeCost node_cost_obj(*subtree, approx_costs.first, approx_costs.second);
    nodes_to_visit.push(node_cost_obj);

    while (!nodes_to_visit.empty()) {
        // Get next node
        current = nodes_to_visit.top();
        nodes_to_visit.pop();

        //Decide if should prune or not, i.e. does hard constraint hold (cost <= delta_remaining)
        if (current.max_approx_cost <= delta_remaining) {
            // Deduct approximation cost of this node. Truth is within max_approx_cost radius of mean contribution
            delta_remaining -= current.max_approx_cost;

            // Add approximated wKDE contribution of this node. (i.e., Wt * K_mean)
            contribution = (current.node->total_weight) * gauss_kern(query_point.begin(), current.node->mean_subtree_point.begin(), bw, this->dim);
            wkde_total += contribution;

            // Update accumulated error distances
            accumulated_lower_error_dist += current.kmin_approx_cost;
            accumulated_upper_error_dist += current.kmax_approx_cost;
        } else {
            // Add true wKDE contribution of this node
            wkde_total += (current.node->weight) * gauss_kern(query_point.begin(), current.node->data_point.begin(), bw, this->dim);

            //Add children to priority queue
            if (current.node->left != NULL) {
                approx_costs = current.node->left->get_approximation_costs(query_point, bw, this->dim);
                dist = by_dist ? L2_distance(current.node->left->data_point, query_point) : 0.0;
                node_cost_obj.reset(*(current.node->left), approx_costs.first, approx_costs.second, dist);
                nodes_to_visit.push(node_cost_obj);
            }
            if (current.node->right != NULL) {
                approx_costs = current.node->right->get_approximation_costs(query_point, bw, this->dim);
                dist = by_dist ? L2_distance(current.node->left->data_point, query_point) : 0.0;
                node_cost_obj.reset(*(current.node->right), approx_costs.first, approx_costs.second, dist);
                nodes_to_visit.push(node_cost_obj);
            }
        }
    }

    double lower = wkde_total - accumulated_lower_error_dist;
    double upper = wkde_total + accumulated_upper_error_dist;

    // Returns the wKDE estimate and an error interval that traps the true value (and should have width not exceeding delta)
    return make_pair(wkde_total, make_pair(lower, upper));
}

vector<double> KDTree::KDE_eval(const vector<vector<double> > &query_points, double bw, double delta_exclude, double delta_mean){
    vector<double> out(query_points.size());
    for(int i=0;i<query_points.size();i++){
        //cout << "Q: " << query_points[i][0] << " " << query_points[i][1] << " " << query_points[i][2] << "\n"; 
        out[i] = this->root->KDE_point_eval(query_points[i],bw,delta_exclude,delta_mean,this->dim);
    }
    return out;
}

vector<pair<double, pair<double, double> > > KDTree::KDE_eval(const vector<vector<double> > &query_points, double bw, double delta, bool dist_based) {
    vector<pair<double, pair<double, double> > > out(query_points.size());
    for (int i = 0; i < query_points.size(); i++) {
        //cout << i << " Q: " << query_points[i][0] << " " << query_points[i][1] << " " << query_points[i][2] << endl; 
        out[i] = this->pruned_kde_point_eval(query_points[i], this->root, bw, delta, dist_based);
    }
    return out;
}

