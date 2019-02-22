#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>
//#include <tuple>
#include "kdtree.h"
#include "kernel_help.h"

#define ONEK 1000 // debugging purposes

using namespace std;

vector<double> read_tab_file(const char *file) {
    ifstream infile(file);
    double a;
    vector<double> out;

    while (infile >> a) {
        out.push_back(a);    
    }

    return out;
}

double mean_diff(vector<double> a, vector<double> b){
    double val = 0;
    for(int i=0; i<(int)a.size(); i++){
        val += abs(a[i]-b[i])/a[i];
    }
    val /= (double)a.size();
    return val;
}

pair<vector<double>, double> prune_KDE(KDTree kd, 
                                       vector< vector<double> > query_points, 
                                       double bw, double delta_exclude, double delta_mean){
    time_t start, end;
    time(&start);
    vector<double> res = kd.KDE_eval(query_points, bw, delta_exclude, delta_mean);
    time(&end);
    return make_pair(res,difftime(end,start));
}

pair<vector<pair<double, pair<double,double> > >, double> interval_KDE(KDTree kd, vector<vector<double> > query_points, double bw, double int_delta, bool dist_based=false){
    time_t start, end;
    time(&start);
    vector<pair<double, pair<double,double> > > res = kd.KDE_eval(query_points, bw, int_delta, dist_based);
    time(&end);
    return make_pair(res,difftime(end,start));
}

pair<vector<double>, double> naive_KDE(vector<double> data_set, vector<double> query, double bw, int dim, vector<double> *weights=NULL){
    vector<double> out((int)(query.size()/dim),0); // results
    double kern_val;
    time_t start, end;
    vector<double>::iterator i,j,w;
    
    time(&start);
    if(weights!=NULL){
        w=weights->begin();
    }

    for(i=data_set.begin(); i!=data_set.end(); i+=dim){
        int o=0;
        for(j=query.begin(); j!=query.end(); j+=dim){
            kern_val = gauss_kern(i,j,bw,dim);
            if(weights!=NULL){
                out[o] += kern_val*(*w);
                w+=1;
            } else {
                out[o] += kern_val*(dim/((double)data_set.size()));
            }
            o = o+1;
        }
    }
    
    time(&end);
    
    return make_pair(out,difftime(end,start));
}

void run_experiments(char *input_file, char *output_file, int dim, double bw, vector<double> deltas){
    KDTree kd;
    char buffer[200], buffer1[200];
    ofstream outfile, outfile1;
    vector<double> points = read_tab_file(input_file);
    kd.build(points, dim, true);
    vector<vector<double> > query_points = kd.convert_data_points(points, dim, true);
    vector<double> points_1k(points.begin(),points.begin()+dim*1E3);
    vector<vector<double> > query_points_1k;
    for(int j=0; j<ONEK; j++){
         query_points_1k.push_back(query_points[j]);
    }
    
    vector<double> naive_times(deltas.size(),0);
    vector<double> interval_times(deltas.size(),0);
    vector<vector<double> > prune_times;
    vector<vector<double> > prune_accus;
    vector<double> prune_time(deltas.size(),0);
    vector<double> prune_accu(deltas.size(),0);
    vector<double> interval_accu(deltas.size(),0);
    
    pair<vector<double>, double> res_time;
    pair<vector<pair<double, pair<double,double> > >, double> res_int_time;
    vector<double> res, res_naive;
    vector<pair<double, pair<double,double> > > res_int;
    double time;
    
    // naive stats
    sprintf(buffer, "%s.naive.result.txt", output_file);
    cout << buffer << "\n";
    res_time = naive_KDE(points, points_1k, bw, dim);
    time = (double)res_time.second/ONEK;
    res = res_time.first;
    outfile.open(buffer);
    outfile << res[0];
    for(int i=1; i<res.size(); i++){
        outfile << "\t" << res[i];
    }
    outfile << "\n";
    outfile.close();
    sprintf(buffer, "%s.naive.time.txt", output_file);
    outfile.open(buffer);
    outfile << time << "\n";
    outfile.close();
    res_naive = res;
    
    // prune stats
    sprintf(buffer, "%s.prune.time.txt", output_file);
    sprintf(buffer1, "%s.prune.accu.txt", output_file);
    cout << buffer << "\n";
    outfile.open(buffer);
    outfile1.open(buffer1);
    for(int i=0; i<deltas.size(); i++){
        for(int j=0; j<deltas.size(); j++){
            cout << "deltas- " << deltas[i] << ", " << deltas[j] << ": ";
            res_time = prune_KDE(kd, query_points_1k, bw, deltas[i], deltas[j]);
            outfile << (j==0 ? "" : "\t") << (double)res_time.second/ONEK;
            outfile1 << (j==0 ? "" : "\t") << mean_diff(res_naive,res_time.first);
            cout << " " << (double)res_time.second/ONEK;
            cout << " " << mean_diff(res_naive,res_time.first) << "\n";
        }
        outfile << "\n";
        outfile1 << "\n";
    }
    outfile.close();
    outfile1.close();
    
    // interval comp stats
    sprintf(buffer, "%s.int.comp.time.txt", output_file);
    sprintf(buffer1, "%s.int.comp.accu.txt", output_file);
    cout << buffer << "\n";
    outfile.open(buffer);
    outfile1.open(buffer1);
    for(int i=0; i<deltas.size(); i++){
        cout << "delta: " << deltas[i] << "\n";
        res_int_time = interval_KDE(kd, query_points_1k, bw, deltas[i]);
        outfile << (i==0 ? "" : "\t") << (double)res_int_time.second/ONEK;
        res.clear();
        for(int j=0; j<res_int_time.first.size(); j++){
            res.push_back(res_int_time.first[j].first);
            //cout << "(" << res_int_time.first[j].first << ")[" << res_int_time.first[j].second.first << "," << res_int_time.first[j].second.second << "]";
        }
        outfile1 << (i==0 ? "" : "\t") << mean_diff(res_naive,res);
        cout << " " << (double)res_int_time.second/ONEK;
        cout << " " << mean_diff(res_naive,res) << "\n";
    }
    outfile.close();
    outfile1.close();
    
    // interval dist stats
    sprintf(buffer, "%s.int.dist.time.txt", output_file);
    sprintf(buffer1, "%s.int.dist.accu.txt", output_file);
    cout << buffer << "\n";
    outfile.open(buffer);
    outfile1.open(buffer1);
    for(int i=0; i<deltas.size(); i++){
        cout << "delta: " << deltas[i];
        res_int_time = interval_KDE(kd, query_points_1k, bw, deltas[i], true);
        outfile << (i==0 ? "" : "\t") << (double)res_int_time.second/ONEK;
        res.clear();
        for(int j=0; j<res_int_time.first.size(); j++){
            res.push_back(res_int_time.first[j].first);
            //cout << "(" << res_int_time.first[j].first << ")[" << res_int_time.first[j].second.first << "," << res_int_time.first[j].second.second << "]\n";
        }
        outfile1 << (i==0 ? "" : "\t") << mean_diff(res_naive,res);
        cout << " " << (double)res_int_time.second/ONEK;
        cout << " " << mean_diff(res_naive,res) << "\n";
    }
    outfile.close();
    outfile1.close();
}

int main(int argc, char* argv[]) { /* input_file output_location dims */
    char* output_file = argv[2];//"../Data/norm2d";
    char buffer[200];
    vector<double> deltas;
    //deltas.push_back(0.0); //this acutally tends to take pretty long ...
    deltas.push_back(1E-12); deltas.push_back(1E-11); deltas.push_back(1E-10);
    deltas.push_back(1E-9); deltas.push_back(1E-8); deltas.push_back(1E-7);
    deltas.push_back(1E-6); deltas.push_back(1E-5); deltas.push_back(1E-4);
    deltas.push_back(1E-3); deltas.push_back(1E-2); deltas.push_back(1E-1);
    
    vector<double> BWs;
    BWs.push_back(0.0625); /*BWs.push_back(0.125);*/ BWs.push_back(0.25); 
    
    cout << "Input: [" << argv[1] << "] Ouput: [" << argv[2] << "] Dims: [" << atoi(argv[3]) << "]\n";
    for(int i=0; i<BWs.size(); i++){
        sprintf(buffer, "%s.%.0f", output_file, BWs[i]*(1E4));
        cout << buffer << endl;
        run_experiments(argv[1],buffer,atoi(argv[3]),BWs[i], deltas);
    }
    
    return 0;
}


