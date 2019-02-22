#include <vector>
#include <cmath>
#define PI 3.14159265359

double gauss_kern(std::vector<double>::iterator X, std::vector<double>::iterator Y, double bw, int dim){
    double kern_val = 1;
    double two_bw_sq = 2*pow(bw,2.0);
    for(int d=0; d<dim; d++){
        kern_val *= exp(-pow(*X-*Y,2.0)/two_bw_sq);
        X = (++X);
        Y = (++Y);
    }
    kern_val /= pow(bw*sqrt(2*PI),dim); 
    return kern_val;
}

