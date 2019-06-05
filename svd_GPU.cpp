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

#include "svd.cu"


int main () {
    SVD svd;

    cout << "begin SVD...\n";

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