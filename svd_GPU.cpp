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

#include <cusolverSp.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include "svd.cu"


int main () {
    SVD svd;

    cout << "begin SVD...\n";

    // Load the training data
    auto start = high_resolution_clock::now();
    svd.load_data();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop-start);
    cout << "loading data took: " << duration.count() << endl;

    // Set the hyperparameters for SVD
    // (k, learning rate, regulariation, num_epochs)
    svd.set_values(50, 0.05, 0.01, 0.0001, 100);

    // Train the model
    start = high_resolution_clock::now();
    svd.train_model();
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop-start);
    cout << "training model took: " << duration.count() << endl;

    return 0;
}