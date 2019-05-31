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

#include "svd.hpp"

using namespace std;
using namespace Eigen;
using namespace std::chrono;

const int NUM_USERS = 458293;
const int NUM_MOVIES = 17770;

const int NUM_USERS_TINY = 13;
const int NUM_MOVIES_TINY = 3;

const int NUM_USERS_SMALL = 10000;

// generator:
struct c_unique {
  int current;
  c_unique() {current=0;}
  int operator()() {return current++;}
} UniqueNumber;

void SVD::set_values (int k, double et, double r, double ep,
                      double max_ep) {
    M = NUM_USERS_SMALL;  // TODO
    N = NUM_MOVIES; // TODO
    K = k;
    eta = et;
    reg = r;
    eps = ep;
    max_epochs = max_ep;

    cout << "done setting values\n";
}

void SVD::load_data() {
    // TRAINING ON BASE (96% OF ALL TRAINING DATA)

    cout << "load training data \n";

    // TODO: resize
    // allocate space for matrix of values
    // Y.resize(NUM_USERS_TINY, NUM_MOVIES_TINY);
    Y.resize(NUM_USERS_SMALL, NUM_MOVIES);

    // TODO: load correct data
    //ifstream file("../data/Archive/tiny.txt"); // actually base.txt
    ifstream file("Archive/small_train.txt");

    cout << "made it here \n";
    int count = 0;
    int uid, mid, date, rating;
    while (file >> uid >> mid >> date >> rating) {
        points.push_back(T(uid-1, mid-1, rating));
        count++;
    }

    cout << "made it here 2\n";

    // convert vector to matrix
    Y.setFromTriplets(points.begin(), points.end());

    // print results
    // cout << MatrixXd(Y) << "\n";
}

void SVD::load_valid() {
    //  CROSS VALIDATING ON VALID AND HIDDEN (4% OF ALL TRAINING DATA)

    cout << "load validation data\n";

    // TODO: load correct data
    //ifstream file("../data/Archive/tiny_qual.txt"); // actually qual.txt
    ifstream file("Archive/small_probe.txt");

    int uid, mid, date, rating;
    while (file >> uid >> mid >> date >> rating) {
        val_uid.push_back(uid-1);
        val_mid.push_back(mid-1);
        val_ratings.push_back(rating);
    }
}

double SVD::get_err() {
    auto start = high_resolution_clock::now();

    cout << "compute error\n";

    double squared_err = 0.0;

    for (int k=0; k<Y.outerSize(); ++k) {
        for (sp_mat::InnerIterator it(Y, k); it; ++it) {
            int Y_ij = it.value();
            int i = it.row();   // row index
            int j = it.col();   // col index (here it is equal to k)
            // it.index(); // inner index, here it is equal to it.row()

            squared_err += 0.5 * pow((Y_ij - U.row(i).dot(V.row(j))), 2);
        }
    }

    squared_err = squared_err / Y.outerSize();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "computed error: " << squared_err << " ; time taken: "<<duration.count()<<endl;;

    return squared_err;
}

VectorXd SVD::grad_U(VectorXd U_i, VectorXd V_j, int Y_ij) {
    //cout << "begin grad U\n";
    double prod = U_i.adjoint() * V_j;
    double Y_ij_d = (double) Y_ij;
    VectorXd sq_err = V_j * (Y_ij_d - prod);
    VectorXd gradient = (U_i * reg - sq_err) * eta;

    return gradient;
}

VectorXd SVD::grad_V(VectorXd U_i, VectorXd V_j, int Y_ij) {
    //cout << "begin grad V\n";
    double prod = U_i.adjoint() * V_j;
    double Y_ij_d = (double) Y_ij;
    VectorXd sq_err = U_i * (Y_ij_d - prod);
    VectorXd gradient = (V_j * reg - sq_err) * eta;

    return gradient;
}

void SVD::train_model() {
    // U = M x K, V = N x K
    // rating Y_ij is approximatedby (UV^T)_ij

    cout << "begin training\n";

    // initialize the entries of U and V to be small random numbers
    U = MatrixXd::Random(M, K);
    V = MatrixXd::Random(N, K);

    double err0 = get_err();
    double err = err0;
    double err1 = 0.0;

    // TODO: resize
    //vector<int> indices(NUM_USERS_TINY);
    //indices.reserve(NUM_USERS_TINY);
    vector<int> indices(NUM_USERS_SMALL);
    indices.reserve(NUM_USERS_SMALL);

    // iota (indices.begin(), indices.end(), 0); // < it dont werk
    generate (indices.begin(), indices.end(), UniqueNumber);

    // continue for max_epochs
    for (int e=0; e<max_epochs; ++e) {
        cout << "begin epoch" << e << "\n";

        auto start = high_resolution_clock::now();

        // shuffle the points in the indices vector
        random_shuffle(indices.begin(), indices.end());

        sp_mat Y_perm = Y;
        cout << "completed permutation\n";

        // update U and V
        for (vector<int>::iterator it=indices.begin(); it!=indices.end(); ++it) {
            int k = *it;

            for (sp_mat::InnerIterator it(Y_perm, k); it; ++it) {
                int Y_ij = it.value();
                int i = it.row();   // row index
                int j = it.col();   // col index (here it is equal to k)
                // it.index(); // inner index, here it is equal to it.row()

                // update U
                VectorXd gradu = grad_U(U.row(i), V.row(j), Y_ij);
                VectorXd urow = U.row(i);
                U.row(i) = urow - gradu;

                //cout << "completed grad U\n";

                // update V
                VectorXd gradv = grad_V(U.row(i), V.row(j), Y_ij);
                VectorXd vrow = V.row(j);
                V.row(j) = vrow - gradv;

                //cout << "completed grad V\n";
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

void SVD::predict_valid() {
    cout << "predict probe\n";

    double squared_err = 0.0;

    for (int i=0; i<val_uid.size(); ++i) {
        int uid = val_uid.at(i);
        int mid = val_mid.at(i);
        double rating = (double) val_ratings.at(i);

        VectorXd urow = U.row(uid);
        VectorXd vrow = V.row(mid);

        double prediction = urow.adjoint() * vrow;

        val_predictions.push_back(rating);

        squared_err += 0.5 * pow((rating - prediction), 2);
    }

    squared_err = squared_err / val_uid.size();

    cout << "completed predictions for probe set, e_out = " << squared_err << "\n";

    // TODO: correct file
    // yeet it out to a file
    //ofstream of("../results/tiny_svd.txt");
    ofstream of("svd_val_results.txt");
    ostream_iterator<double> output_iterator(of, "\n");
    copy(val_predictions.begin(), val_predictions.end(), output_iterator);
}

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

    svd.load_valid();

    start = high_resolution_clock::now();
    svd.predict_valid();
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop-start);
    cout << "predicting valid took: " << duration.count() << endl;

    return 0;
}