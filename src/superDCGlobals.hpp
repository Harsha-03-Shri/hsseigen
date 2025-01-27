#pragma once
#include<bits/stdc++.h>
#include<string.h>
#include "BinTree.h"
#include "superDC.h"
#include "divide2.h"
#include "superdcmv_desc.h"
#include "superdcmv_cauchy.h"
#include "eigenmatrix.h"
#include "secular.h"
#include "Generators.h"
#include <sys/time.h>

#if defined(PARALLEL) || defined(PARALLEL_TASK)
#include "omp.h"
#endif

#ifdef DIST
// extern "C"{
#include <mpi.h>
// }
#endif

extern "C"
{
    #include<lapacke.h>
    #include<lapack.h>
    #include<cblas.h>
}


int N;
DVD *resDvd;
EIG_MAT **Q0;
std::pair<int, int>* q0Sizes;
double **Lam; 
int *LamSizes;
BinTree *bt;
std::pair<int,int>* l;

void Eig_func(int i){
    vector<int> ch = bt->GetChildren(i + 1);
    
    if(ch.empty())
    {
        std::pair<double*, double *> E = computeLeafEig(make_pair(resDvd->dSizes[i].first, resDvd->dSizes[i].second), resDvd->D[i], i);
        Lam[i] = E.first;         
        LamSizes[i] = resDvd->dSizes[i].first;

        Q0[i] = new EIG_MAT();
        Q0[i]->Q0_leaf = E.second;            
        Q0[i]->Q0_nonleaf = NULL;
        q0Sizes[i] = {resDvd->dSizes[i].first, resDvd->dSizes[i].second};
       // light_switch[i] = true;
       return;
    }
    
    int left = ch[0];
    int right = ch[1];

    Eig_func(left-1);
    Eig_func(right-1);   
    
    //compute
    superdcmv_desc(Q0,q0Sizes,(resDvd->Z[i]),resDvd->zSizes[i],bt,i,1,l,fmmTrigger);           
    Lam[i] = new double[(LamSizes[left-1]) + (LamSizes[right-1])];

    std::copy(Lam[left-1], Lam[left-1] + LamSizes[left-1], Lam[i]);
    std::copy(Lam[right-1], Lam[right-1] + LamSizes[right-1], Lam[i] + LamSizes[left - 1]);
            
    LamSizes[i] = (LamSizes[left - 1]) + (LamSizes[right - 1]);
    //std::sort(Lam[i], Lam[i]+LamSizes[i]);

    delete [] Lam[left - 1];
    delete [] Lam[right - 1];

    LamSizes[left - 1]  = 0;
    LamSizes[right - 1] = 0;

    int r             = resDvd->zSizes[i].second;

    Q0[i]             = new EIG_MAT();
    Q0[i]->Q0_leaf    = NULL;

    nonleaf **n_leaf    = new nonleaf*[r];
            
    std::pair<double *, nonleaf**> result = r_RankOneUpdate(Lam[i], LamSizes[i], resDvd->zSizes[i], resDvd->Z[i], n_leaf, r);
    Lam[i] = result.first;
    Q0[i]->Q0_nonleaf = result.second;
    Q0[i]->n_non_leaf = r;
    q0Sizes[i] = {1, r};
    return;
}






std::pair<double *, nonleaf**> r_RankOneUpdate(double* Lam, int lamSize, std::pair<int, int>zSize, double* Z, nonleaf **n_leaf, int r){
    
    std::pair<double *, nonleaf**> result;
    
    double *temp_d   = Lam;
    int temp_d_size = lamSize;
    for(int j = 0; j < r; j++)
    {
                          
        double *tempZ = new double[zSize.first];

        for(int row = 0; row < zSize.first; row++){
            tempZ[row] = Z[j + row * r];
        }

        SECU *res_sec;
        res_sec = secular(temp_d, temp_d_size, tempZ, zSize.first,  fmmTrigger);
        n_leaf[j] = res_sec->Q;
                
        delete [] temp_d;
        temp_d = res_sec->Lam;
                
                
        if(j < (r-1))
        {
            double *tempZi = new double[zSize.first * (r - (j + 1))];

            for(int row = 0; row < zSize.first; row++)
                memcpy(tempZi + row*(r-(j+1)), Z + (j + 1) + row * (zSize.second), sizeof(double) * (r - (j + 1)) );
                    
            tempZi = superdcmv_cauchy((n_leaf[j]), {1, 7}, tempZi, {zSize.first, (r- ( j + 1)) }, 1, fmmTrigger);

            for(int row = 0; row < zSize.first; row++)
                memcpy(Z + j + 1 + row * (zSize.second), tempZi + row*(r - (j + 1)), sizeof(double) * (r - (j + 1)));
                    
            delete [] tempZi;
        }
                
    delete [] tempZ;
                //will add rho later
    }
    
    result.first = temp_d;
    result.second = n_leaf;
    return result;       
}



std::pair<double*, double*> computeLeafEig(std::pair<int, int> dSize, double *D, int i) {
    std::pair<double*, double*> result;
    int n = dSize.first;
    int lda = dSize.second;

    double *E = new double[n];
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    double *d_D, *d_E;
    int *d_info;
    int lwork = 0;
    double *d_work;

    cudaMalloc((void**)&d_D, n * lda * sizeof(double));
    cudaMalloc((void**)&d_E, n * sizeof(double));
    cudaMalloc((void**)&d_info, sizeof(int));

    double *D_transposed = new double[n * lda];
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            D_transposed[col * lda + row] = D[row * lda + col];
        }
    }
    cudaMemcpy(d_D, D_transposed, n * lda * sizeof(double), cudaMemcpyHostToDevice);
 

    cusolverDnDsyevd_bufferSize(handle, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER, n, d_D, lda, nullptr, &lwork);
    cudaMalloc((void**)&d_work, lwork * sizeof(double));

    int info;
    cusolverDnDsyevd(handle, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER, n, d_D, lda, d_E, d_work, lwork, d_info);

    cudaMemcpy(E, d_E, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    if (info > 0) {
        std::cerr << "Eigensolver failed for node: " << (i + 1) << "\n";
         cusolverDnDestroy(handle);
        cudaFree(d_D);
        cudaFree(d_E);
        cudaFree(d_info);
        exit(1);
    }
    result.first = E;
    delete[] D_transposed;

    cudaFree(d_D);
    cudaFree(d_E);
    cudaFree(d_work);
    cudaFree(d_info);

    cusolverDnDestroy(handle);

    return result;
}
