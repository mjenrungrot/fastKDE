#ifndef KERN_H
#define KERN_H

#include <cmath>
#include <vector>

double gauss_kern(vector<double>::iterator X, vector<double>::iterator Y, double bw, int dim);

#endif
