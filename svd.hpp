#ifndef SVD_H
#define SVD_H

#include <vector>
#include <Eigen/Sparse>
#include <stdint.h>
#include <functional>

using namespace std;
using namespace Eigen;

// declares a row-major sparse matrix type of double
typedef SparseMatrix<double, RowMajor> sp_mat; 

// row index, column index, value
typedef Triplet<double> T;

class SVD {
    int M, N, K;
    double eta, reg, eps, max_epochs;

    sp_mat Y; // NUM_USERS x NUM_MOVIES
    // vector<NUM_USERS_SMALL, NUM_MOVIES> Y; 
    vector<T> points;

    MatrixXd U; // NUM_USERS X K
    MatrixXd V; // NUM_MOVIES X K

    vector<int> val_uid;
    vector<int> val_mid;
    vector<int> val_ratings;

    vector<double> val_predictions;

  public:
    void set_values(int, double, double, double, double);

    void load_data();

    void load_valid();

    double get_err();

    VectorXd grad_U(VectorXd, VectorXd, int);

    VectorXd grad_V(VectorXd, VectorXd, int);

    void train_model();

    void predict_valid();

};

#endif //SVD_H