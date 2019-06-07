#ifndef SVD_H
#define SVD_H

#include <vector>
#include <stdint.h>
#include <functional>

#include <cusolverSp.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

const int NUM_USERS = 458293;

const int NUM_USERS_TINY = 13;
const int NUM_MOVIES_TINY = 3;

const int NUM_USERS_SMALL = 10000;
const int NUM_MOVIES = 17770;

class SVD {
    int M, N, K;
    double eta, reg, eps, max_epochs;

    // declare all the scary matrices
    double * u_mat;
    double * v_mat;

    // Initialize cublas handle
    cublasHandle_t handle;

    vector<vector<double>> Y; // NUM_USERS x NUM_MOVIES
    
    // this should work but it isn't
    // vector<vector<double>> Y (NUM_USERS_SMALL, vector<double>(NUM_MOVIES, 0));

    vector<vector<double>> U; // NUM_USERS X K
    vector<vector<double>> V; // NUM_MOVIES X K

    vector<int> val_uid;
    vector<int> val_mid;
    vector<int> val_ratings;

    vector<double> val_predictions;

  public:
    void set_values(int, double, double, double, double);

    void load_data();

    void load_valid();

    double get_err();

    vector<double> grad_U(vector<double>, vector<double>, int);

    vector<double> grad_V(vector<double>, vector<double>, int);

    vector<double> sub_vectors(vector<double>, vector<double>);

    void train_model();

    void predict_valid();
};

#endif //SVD_H