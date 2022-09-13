#ifndef CAUCHYLIKEMATVEC_H
#define CAUCHYLIKEMATVEC_H
#include <utility>
double *cauchylikematvec(double **Qcd, std::pair<int,int>*qcSizes, const int *org, double *Xx, std::pair<int,int>xSize,int ifTrans, double N=1024);
#endif