#ifndef SUPERDCMV_H
#define SUPERDCMV_H
#include<utility>
#include"BinTree.h"
#include"eigenmatrix.h"
double* superdcmv(EIG_MAT **Qt, std::pair<int, int>*qSize, double *x, BinTree* bt, int ifTrans,double N);
#endif
