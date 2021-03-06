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
#include <chrono>
#include <omp.h>

#include "svd.cuh"

using namespace std;
using namespace std::chrono;

// generator:
struct c_unique {
  int current;
  c_unique() {current=0;}
  int operator()() {return current++;}
} UniqueNumber;

// Set the hyperparameters for SVD
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

// Load the training data into a two dimensional vector
void SVD::load_data() {

    cout << "load training data \n";

    // Initialize handle
    cublasCreate(&handle);

    ifstream file("Archive/small_train.txt");

    // stupid way of initializing for now
    for (int i = 0; i < NUM_USERS_SMALL; ++i)
    {
        vector<double> temp(NUM_MOVIES, 0);
        Y.push_back(temp);
    }

    int uid, mid, date, rating;
    while (file >> uid >> mid >> date >> rating) {
        Y[uid-1][mid-1] = rating;
    }
}

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
            vector<double> vrow = V[j];

            // DONE: Convert to cuBLAS
            // squared_err += 0.5 * pow((Y_ij - urow.dot(vrow)), 2);

            double * dev_urow;
            double * dev_vrow;

            // Convert from vector to c array
            double * urow_array = &urow[0];
            double * vrow_array = &vrow[0];

            cout << "finished converting\n";

            // Allocate device memory for the vectors
            cudaMalloc((void **) &dev_urow, K * sizeof(double));

            cudaMalloc((void **) &dev_vrow, K * sizeof(double));

            cout << "we have memory\n";

            // Copy arrays from the host to the device
            cudaMemcpy(dev_urow, urow_array, sizeof(double) * K, cudaMemcpyHostToDevice);
            cudaMemcpy(dev_vrow, vrow_array, sizeof(double) * K, cudaMemcpyHostToDevice);

            cout << "we copied memory\n";

            // Compute the dot product of urow and vrow
            double * dev_dot;
            cudaMalloc((void **) &dev_dot, sizeof(double));

            cout << "about to take the dot\n";

            cublasDdot(handle, K, dev_urow, 1, dev_vrow, 1, dev_dot);

            cout << "we took the dot product\n";

            // Copy the result from device to host machine
            double dot;
            cudaMemcpy(&dot, dev_dot, sizeof(double), cudaMemcpyDeviceToHost);

            squared_err += 0.5 * pow((Y_ij - dot), 2);

            // Free the cuda memory 
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
    double * vrow_array = &V_j[0];

    // Allocate device memory for the vectors
    cudaMalloc((void **) &dev_urow, K * sizeof(double));

    cudaMalloc((void **) &dev_vrow, K * sizeof(double));  

    // Copy arrays from the host to the device
    cudaMemcpy(dev_urow, urow_array, sizeof(double) * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vrow, vrow_array, sizeof(double) * K, cudaMemcpyHostToDevice);

    // Compute the dot product of urow and vrow
    double * dev_dot;
    cudaMalloc((void **) &dev_dot, sizeof(double));
    cublasDdot(handle, K, dev_urow, 1, dev_vrow, 1, dev_dot);  

    // Copy the result from device to host machine
    double dot;
    cudaMemcpy(&dot, dev_dot, sizeof(double), cudaMemcpyDeviceToHost);

    double Y_ij_d = (double) Y_ij;

    // sq_err = V_j * (Y_ij_d - prod)
    double alpha = -1 * Y_ij_d - dot;
    cublasDscal(handle, K, &alpha, dev_vrow, 1); // dev_vrow now contains -sq_err

    // gradient = (U_i * reg - sq_err) * eta;

    // dev_urow now contains U_i * reg
    cublasDscal(handle, K, &reg, dev_urow, 1); 

    // dev_vrow now contains (U_i * reg) - sq_err
    double one = 1.0;

    cublasDaxpy(handle, K, &one, dev_urow, 1, dev_vrow, 1);

    // dev_vrow now contains (U_i * reg - sq_err) * eta
    cublasDscal(handle, K, &eta, dev_vrow, 1); 

    double * gradient_array;
    cudaMalloc((void **) &gradient_array, K * sizeof(double));
    // Copy over the data from the device
    cudaMemcpy(gradient_array, dev_vrow, sizeof(double) * K, cudaMemcpyDeviceToHost);

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
    double * vrow_array = &V_j[0];

    // Allocate device memory for the vectors
    cudaMalloc((void **) &dev_urow, K * sizeof(double));

    cudaMalloc((void **) &dev_vrow, K * sizeof(double));  

    // Copy arrays from the host to the device
    cudaMemcpy(dev_urow, urow_array, sizeof(double) * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vrow, vrow_array, sizeof(double) * K, cudaMemcpyHostToDevice);

    // Compute the dot product of urow and vrow
    double * dev_dot;
    cudaMalloc((void **) &dev_dot, sizeof(double));
    cublasDdot(handle, K, dev_urow, 1, dev_vrow, 1, dev_dot);  

    // Copy the result from device to host machine
    double dot;
    cudaMemcpy(&dot, dev_dot, sizeof(double), cudaMemcpyDeviceToHost);

    double Y_ij_d = (double) Y_ij;

    // sq_err = U_i * (Y_ij_d - prod)
    double alpha = -1 * Y_ij_d - dot;
    cublasDscal(handle, K, &alpha, dev_urow, 1); // dev_urow now contains -sq_err

    // gradient = (V_j * reg - sq_err) * eta;

    // dev_vrow now contains V_j * reg
    cublasDscal(handle, K, &reg, dev_vrow, 1); 

    // dev_urow now contains V_j * reg - sq_err
    double one = 1.0;

    cublasDaxpy(handle, K, &one, dev_vrow, 1, dev_urow, 1);

    // dev_urow now contains (V_j * reg - sq_errU_i * reg - sq_err) * eta
    cublasDscal(handle, K, &eta, dev_urow, 1); 

    double * gradient_array;
    // Copy over the data from the device
    cudaMalloc((void **) &gradient_array, sizeof(double) * K);
    cudaMemcpy(gradient_array, dev_vrow, sizeof(double) * K, cudaMemcpyDeviceToHost);

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
vector<double> SVD::sub_vectors(vector<double> a, vector<double> b) {

    double * dev_a;
    double * dev_b;

    // multiply b by -1 element-wise to sub instead of add
    for (uint i = 0; i < b.size(); ++i)
    {
        b[i] = b[i] * -1;
    }

    // convert from vector to c array
    double * a_array = &a[0];
    double * b_array = &b[0];

    // Allocate device memory for the vectors
    cudaMalloc((void **) &dev_a, K * sizeof(double));

    cudaMalloc((void **) &dev_b, K * sizeof(double)); 

    // Copy arrays from the host to the device
    cudaMemcpy(dev_a, a_array, sizeof(double) * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b_array, sizeof(double) * K, cudaMemcpyHostToDevice);

    double one = 1.0;
    cublasDaxpy(handle, K, &one, dev_a, 1, dev_b, 1); // result in dev_b

    double * sub_array;
    cudaMalloc((void **) &sub_array, K * sizeof(double));

    // Copy over the data from the device
    cudaMemcpy(sub_array, dev_b, sizeof(double) * K, cudaMemcpyDeviceToHost);

    // convert c array to vector
    vector<double> sub (sub_array, 
            sub_array + sizeof sub_array / sizeof sub_array[0]);

    // Free the cuda memory
    cudaFree(dev_a);
    cudaFree(dev_b);

    return sub;
}

void SVD::train_model() {
    // U = M x K, V = N x K
    // rating Y_ij is approximatedby (UV^T)_ij

    cout << "begin training\n";

    // TODO: Fill feature matrices with random -0.5 5to 0.5 values
    // double lower_bound = -0.5;
    // double upper_bound = 0.5;
    // uniform_real_distribution<double> unif(lower_bound, upper_bound);
    // default_random_engine re;
    // generate(U.begin(), U.end(), unif(re));
    // generate(V.begin(), V.end(), unif(re));

    // For now, just set feature matrices to 0
    for (int i = 0; i < NUM_USERS_SMALL; ++i)
    {
        vector<double> temp(K, 0);
        U.push_back(temp);
    }

    for (int i = 0; i < NUM_MOVIES; ++i)
    {
        vector<double> temp(K, 0);
        V.push_back(temp);
    }

    // Get initial error to use in stopping condition 
    double err0 = get_err();
    double err = err0;
    double err1 = 0.0;

    // Create index vector to shuffle the points after every epoch
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
            // potential problem: no easy way to only iterate through the 
            // non zero values like there was for eigen so it might be slower
            // creating our own GPU compatible version of sparse matrix 
            // would be too complicated.

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
