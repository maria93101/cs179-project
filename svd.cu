#include <string>
#include <numeric>
#include <vector>
#include <iostream>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <assert.h>
#include <functional>
#include <Eigen/Sparse>
#include <chrono>
#include <omp.h>

#include "svd_GPU.hpp"

using namespace std;
using namespace Eigen;
using namespace std::chrono;

// generator:
struct c_unique {
  int current;
  c_unique() {current=0;}
  int operator()() {return current++;}
} UniqueNumber;

void SVD::set_values (int k, double et, double r, double ep,
                      double max_ep) {
    M = NUM_USERS_SMALL;   
    N = NUM_MOVIES;
    K = k;
    eta = et;
    reg = r;
    eps = ep;
    max_epochs = max_ep;

    cout << "done setting values\n";
}


void SVD::initialize () {

    // Allocate device memory for the matrices
    CUDA_CALL(cudaMalloc((void **) &u_mat, 
                    NUM_USERS_SMALL * K * sizeof(double)));

    CUDA_CALL(cudaMalloc((void **) &v_mat, 
                    NUM_MOVIES * K * sizeof(double)));

    // Convert U and V into flat arrays in a gross way

    float * u_flat = new float[NUM_USERS_SMALL * K];
    for (i = 0; i < NUM_USERS_SMALL; i++) {
        copy(U[i].begin(), U[i].end(), u_flat);
        u_flat += U[i].size();
    }    

    float * v_flat = new float[NUM_MOVIES * K];
    for (i = 0; i < NUM_MOVIES; i++) {
        copy(V[i].begin(), V[i].end(), v_flat);
        v_flat += V[i].size();
    }    

    // Copy over the data onto the device
    CUBLAS_CALL(cublasSetMatrix(NUM_USERS_SMALL, K, sizeof(double), 
                        U, NUM_USERS_SMALL, u_mat, NUM_USERS_SMALL));
    
    CUBLAS_CALL(cublasSetMatrix(NUM_MOVIES, K, sizeof(double), 
                        v_flat, NUM_MOVIES, v_mat, NUM_MOVIES));   

    // TODO: do we need to transpose things for row/col major purposes???
    // the default is col major but we have our data in row major??? 

    cout << "done initializing GPU stuff \n";
}

void SVD::load_data() {

    cout << "load training data \n";

    ifstream file("Archive/small_train.txt");

    // stupid way of initializing for now
    for (int i = 0; i < NUM_USERS_SMALL; ++i)
    {
        vector<double> temp(NUM_MOVIES, 0);
        Y.push_back(temp);
    }

    cout << "made it here \n";
    int uid, mid, date, rating;
    while (file >> uid >> mid >> date >> rating) {
        Y[uid-1][mid-1] = rating;
    }

    cout << "made it here 2\n";
}

// void SVD::load_valid() {
//     //  CROSS VALIDATING ON VALID AND HIDDEN (4% OF ALL TRAINING DATA)

//     cout << "load validation data\n";

//     //ifstream file("../data/Archive/tiny_qual.txt"); // actually qual.txt
//     ifstream file("Archive/small_probe.txt");

//     int uid, mid, date, rating;
//     while (file >> uid >> mid >> date >> rating) {
//         val_uid.push_back(uid-1);
//         val_mid.push_back(mid-1);
//         val_ratings.push_back(rating);
//     }
// }

double SVD::get_err() {
    auto start = high_resolution_clock::now();

    cout << "compute error\n";

    double squared_err = 0.0;

    for (auto row = Y.begin(); row != Y.end(); ++row) {
        
        for (auto col = row->begin(); col != row->end(); ++col) {
            int Y_ij = (int) *col; // actual rating value
            int i = distance(Y.begin(), row); // row index
            int j = distance(row->begin(), col); // col index

            cout << "got index\n";
           
            vector<double> urow = U[i];
            vector<double> vrow = V[i];

            // DONE: squared_err += 0.5 * pow((Y_ij - urow.dot(vrow)), 2);

            double * dev_urow;
            double * dev_vrow;

            // Convert from vector to c array
            double * urow_array = &urow[0];
            double * vrow_array = &vrow[0];

            // Allocate device memory for the vectors
            CUDA_CALL(cudaMalloc((void **) &dev_urow, K * sizeof(double)));

            CUDA_CALL(cudaMalloc((void **) &dev_vrow, K * sizeof(double)));

            // Compute the dot product of urow and vrow
            double * dev_dot;
            cublasDdot(handle, K, dev_urow, 1, dev_vrow, 1, dev_dot);

            // Copy the result from device to host machine
            double dot;
            cudaMemcpy(&dot, void *dev_dot, sizeof(double), cudaMemcpyDeviceToHost);

            squared_err += 0.5 * pow((Y_ij - dot), 2);

            // Free the cuda memory 
            // TODO: do we need to use cublasDestroy????
            cudaFree(dev_urow);
            cudaFree(dev_vrow);
        }
    }

    squared_err = squared_err / Y.size();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "computed error: " << squared_err << " ; time taken: "<<duration.count()<<endl;;

    return squared_err;
}

vector<double> SVD::grad_U(vector<double> U_i, vector<double> V_j, int Y_ij) {
    cout << "begin grad U\n";

    // DONE: convert to cuBLAS
    // double prod = U_i.adjoint() * V_j;
    // double Y_ij_d = (double) Y_ij;
    // VectorXd sq_err = V_j * (Y_ij_d - prod);
    // VectorXd gradient = (U_i * reg - sq_err) * eta;

    double * dev_urow;
    double * dev_vrow;

    // convert from vector to c array
    double * urow_array = &U_i[0];
    double * vrow_array = &V_i[0];

    // Allocate device memory for the vectors
    CUDA_CALL(cudaMalloc((void **) &dev_urow, K * sizeof(double)));

    CUDA_CALL(cudaMalloc((void **) &dev_vrow, K * sizeof(double)));  

    // Copy arrays from the host to the device
    cudaMemcpy(dev_urow, urow_array, sizeof(double) * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vrow, vrow_array, sizeof(double) * K, cudaMemcpyHostToDevice);

    // Compute the dot product of urow and vrow
    double * dev_dot;
    cublasDdot(handle, K, dev_urow, 1, dev_vrow, 1, dev_dot);  

    // Copy the result from device to host machine
    double dot;
    cudaMemcpy(&dot, void *dev_dot, sizeof(double), cudaMemcpyDeviceToHost);

    double Y_ij_d = (double) Y_ij;

    // sq_err = V_j * (Y_ij_d - prod)
    double alpha = -1 * Y_ij_d - dot;
    cublasDscal(handle, K, &alpha, dev_vrow, 1); // dev_vrow now contains -sq_err

    // gradient = (U_i * reg - sq_err) * eta;

    // dev_urow now contains U_i * reg
    cublasDscal(handle, K, &reg, dev_urow, 1); 

    // dev_vrow now contains (U_i * reg) - sq_err
    cublasDaxpy(handle, K, 1.0, dev_urow, 1, dev_vrow, 1);

    // dev_vrow now contains (U_i * reg - sq_err) * eta
    cublasDscal(handle, K, &eta, dev_vrow, 1); 

    double * gradient_array;
    // Copy over the data from the device
    cudaMemcpy(gradient_array, void *dev_vrow, sizeof(double) * K, cudaMemcpyDeviceToHost);

    // convert c array to vector
    vector<double> gradient (gradient_array, 
            gradient_array + sizeof gradient_array / sizeof gradient_array[0]);

    // Free the cuda memory
    cudaFree(dev_urow);
    cudaFree(dev_vrow);
    cudaFree(dev_dot);

    return gradient;
}

vector<double> SVD::grad_V(vector<double> U_i, vector<double> V_j, int Y_ij) {
    cout << "begin grad V\n";

    // DONE: convert to cuBLAS
    // double prod = U_i.adjoint() * V_j;
    // double Y_ij_d = (double) Y_ij;
    // VectorXd sq_err = U_i * (Y_ij_d - prod);
    // VectorXd gradient = (V_j * reg - sq_err) * eta;

    double * dev_urow;
    double * dev_vrow;

    // convert from vector to c array
    double * urow_array = &U_i[0];
    double * vrow_array = &V_i[0];

    // Allocate device memory for the vectors
    CUDA_CALL(cudaMalloc((void **) &dev_urow, K * sizeof(double)));

    CUDA_CALL(cudaMalloc((void **) &dev_vrow, K * sizeof(double)));  

    // Copy arrays from the host to the device
    cudaMemcpy(dev_urow, urow_array, sizeof(double) * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vrow, vrow_array, sizeof(double) * K, cudaMemcpyHostToDevice);

    // Compute the dot product of urow and vrow
    double * dev_dot;
    cublasDdot(handle, K, dev_urow, 1, dev_vrow, 1, dev_dot);  

    // Copy the result from device to host machine
    double dot;
    cudaMemcpy(&dot, void *dev_dot, sizeof(double), cudaMemcpyDeviceToHost);

    double Y_ij_d = (double) Y_ij;

    // sq_err = U_i * (Y_ij_d - prod)
    double alpha = -1 * Y_ij_d - dot;
    cublasDscal(handle, K, &alpha, dev_urow, 1); // dev_urow now contains -sq_err

    // gradient = (V_j * reg - sq_err) * eta;

    // dev_vrow now contains V_j * reg
    cublasDscal(handle, K, &reg, dev_vrow, 1); 

    // dev_urow now contains V_j * reg - sq_err
    cublasDaxpy(handle, K, 1.0, dev_vrow, 1, dev_urow, 1);

    // dev_urow now contains (V_j * reg - sq_errU_i * reg - sq_err) * eta
    cublasDscal(handle, K, &eta, dev_urow, 1); 

    double * gradient_array;
    // Copy over the data from the device
    cudaMemcpy(gradient_array, void *deu_vrow, sizeof(double) * K, cudaMemcpyDeviceToHost);

    // convert c array to vector
    vector<double> gradient (gradient_array, 
            gradient_array + sizeof gradient_array / sizeof gradient_array[0]);

    // Free the cuda memory
    cudaFree(dev_urow);
    cudaFree(dev_vrow);
    cudaFree(dev_dot);

    return gradient;
}

// Take in two vectors of doubles, convert to arrays in host memory, and 
// perform a - b. Copy array to host memory and return in vector form.

void SVD::sub_vectors(vector<double> a, vector<double> b) {

    double * dev_a;
    double * dev_b;

    // multiply b by -1 element-wise
    for (int i = 0; i < b.size(); ++i)
    {
        b[i] = b[i] * -1;
    }

    // convert from vector to c array
    double * a_array = &a[0];
    double * b_array = &b[0];

    // Copy arrays from the host to the device
    cudaMemcpy(dev_a, a_array, sizeof(double) * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b_array, sizeof(double) * K, cudaMemcpyHostToDevice);

    cublasDaxpy(handle, K, 1.0, dev_a, 1, dev_b, 1); // result in dev_b

    double * sub_array;
    // Copy over the data from the device
    cudaMemcpy(sub_array, void *dev_b, sizeof(double) * K, cudaMemcpyDeviceToHost);

    // convert c array to vector
    vector<double> sub (sub_array, 
            sub_array + sizeof sub_array / sizeof sub_array[0]);

    // Free the cuda memory
    cudaFree(dev_urow);
    cudaFree(dev_vrow);

    return sub;
}

void SVD::train_model() {
    // U = M x K, V = N x K
    // rating Y_ij is approximatedby (UV^T)_ij

    cout << "begin training\n";

    // fill feature matrices with random 0-0.5 values
    // double lower_bound = -0.5;
    // double upper_bound = 0.5;
    // uniform_real_distribution<double> unif(lower_bound, upper_bound);
    // default_random_engine re;
    // generate(U.begin(), U.end(), unif(re));
    // generate(V.begin(), V.end(), unif(re));

    // for now, just 0
    // TODO: fix, this is really stupid
    for (int i = 0; i < NUM_USERS_SMALL; ++i)
    {
        vector<double> temp;
        for (int j = 0; i < K; ++j)
        {
            temp.push_back(0.0);
        }
        U.push_back(temp);
    }

    for (int i = 0; i < NUM_MOVIES; ++i)
    {
        vector<double> temp;
        for (int j = 0; i < K; ++j)
        {
            temp.push_back(0.0);
        }
        V.push_back(temp);
    }

    // get initial error
    double err0 = get_err();
    double err = err0;
    double err1 = 0.0;

    vector<int> indices(NUM_USERS_SMALL);
    indices.reserve(NUM_USERS_SMALL);

    generate (indices.begin(), indices.end(), UniqueNumber);

    // continue for max_epochs
    for (int e=0; e<max_epochs; ++e) {
        cout << "begin epoch" << e << "\n";

        auto start = high_resolution_clock::now();

        // shuffle the points in the indices vector
        random_shuffle(indices.begin(), indices.end());

        vector<vector<double>> Y_perm = Y;
        cout << "completed permutation\n";


        // update U and V
        for (vector<int>::iterator it=indices.begin(); it!=indices.end(); ++it) {
            int k = *it;

            // potential problem: no easy way to only iterate through the 
            // non zero values like there was for eigen so it might be slower

            for (auto row = Y.begin(); row != Y.end(); ++row)
            {
                for (auto col = row->begin(); col != row->end(); ++col) {
                    int Y_ij = (int) *col; // actual rating value
                    int i = distance(Y.begin(), row); // row index
                    int j = distance(row->begin(), col); // col index

                    // Update U
                    vector<double> gradu = grad_U(U[i], V[j], Y_ij);
                    vector<double> urow = U[i];
                    // DONE: Convert U[i] = urow - gradu to CuBLAS
                    U[i] = sub_vectors(urow, gradu);

                    cout << "completed grad U\n";

                    // Update V
                    vector<double> gradv = grad_V(U[i], V[j], Y_ij);
                    vector<double> vrow = V[i];
                    // DONE: Convert V[i] = vrow - gradv to CuBLAS
                    V[i] = sub_vectors(vrow, gradv);

                    cout << "completed grad V\n";
                }
            }            

        cout << "updated U, V for all points in training set\n";

        double err_prev = err;
        err = get_err();

        if (e == 0) {
            err1 = err;
        }

        // check if error reduction satisfied
        double comp = (err_prev - err) / (err0 - err1);
        cout << "error reduction = " << comp << "\n";
        if (comp <= eps) {
            cout << " YEET\n";
            break;
        }

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "computed updated error for epoch " << e << "; total time taken: "<<duration.count()<<"\n";
    }

    // set final e_in
    double e_in = err;

    cout << "completed training, e_in = " << e_in << "\n";
    }
}

void SVD::predict_valid() {
    // cout << "predict probe\n";

    // double squared_err = 0.0;

    // for (int i=0; i<val_uid.size(); ++i) {
    //     int uid = val_uid.at(i);
    //     int mid = val_mid.at(i);
    //     double rating = (double) val_ratings.at(i);

    //     VectorXd urow = U.row(uid);
    //     VectorXd vrow = V.row(mid);

    //     double prediction = urow.adjoint() * vrow;

    //     val_predictions.push_back(rating);

    //     squared_err += 0.5 * pow((rating - prediction), 2);
    // }

    // squared_err = squared_err / val_uid.size();

    // cout << "completed predictions for probe set, e_out = " << squared_err << "\n";

    // // yeet it out to a file
    // ofstream of("svd_val_results.txt");
    // ostream_iterator<double> output_iterator(of, "\n");
    // copy(val_predictions.begin(), val_predictions.end(), output_iterator);
}

int main () {
    SVD svd;

    cout << "begin SVD...\n";

    // Initialize handle
    CUBLAS_CALL(cublasCreate(&handle));

    auto start = high_resolution_clock::now();
    svd.load_data();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop-start);
    cout << "loading data took: " << duration.count() << endl;

    svd.set_values(50, 0.05, 0.01, 0.0001, 100);

    start = high_resolution_clock::now();
    svd.train_model();
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop-start);
    cout << "training model took: " << duration.count() << endl;

    // svd.load_valid();

    // start = high_resolution_clock::now();
    // svd.predict_valid();
    // stop = high_resolution_clock::now();
    // duration = duration_cast<microseconds>(stop-start);
    // cout << "predicting valid took: " << duration.count() << endl;

    return 0;
}