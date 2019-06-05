
#include<iostream>
#include <fstream>
#include<vector>
#include<string>
#include<map>
#include "data.h"
#include <tuple>
#include<cmath>
#include<algorithm>
#include <cuda_runtime.h>

#include <iterator>
#include <sstream>
#include <chrono>
//Single line comment
using namespace std;
using namespace std::chrono;
// It takes about 5 minutes to run..

float get_error(vector<float> real_ratings, vector<float> ratings)
{
    float summ = 0;
    for (int i = 0; i < 7000; i++)
    {
        summ += pow(ratings[i] - real_ratings[i], 2);
    }
    return sqrt(summ / 7000);
}

void knn(Data *data, float **cij_lib, int alpha, int k, string ifile, string ofile, bool probe)
{
    vector<float> true_rat;
    int counter = 0;
    
    ofstream txtOut;
    txtOut.open (ofile);
    vector<pair<float, float>> paired_movieid;
    vector<int> val_uid;
    vector<int> val_mid;
    vector<int> val_ratings;
    ifstream file(ifile);
    int uid, mid, date, rating;
    float e_out = 0;
    if (probe)
    {
        while (file >> uid >> mid >> date >> rating) {
            val_uid.push_back(uid-1);
            val_mid.push_back(mid-1);
            val_ratings.push_back(rating);
        }
    }
    else
    {
        while (file >> uid >> mid >> date) {
            val_uid.push_back(uid-1);
            val_mid.push_back(mid-1);
            val_ratings.push_back(0);
        }
    }
    for (int i = 0; i < val_uid.size(); i++)
    {
        int user_id = val_uid[i];
        int cur_movie_id = val_mid[i];
        vector<float> user_list_movies = data->user_movie[user_id];
        
        // This could be an array if it doesnt have pair.
        float * cij;
        
        
        ////////////////////////////////////////////////////////
        Data * gpu_data;
        cudaMalloc((void **) &gpu_data, sizeof(Data)));

        float ** gpu_cij_lib;
        
        cudaMalloc((void **) &gpu_cij_lib, 17770 * sizeof(float));
        for (int i = 0; i < 17770; i++)
        {
            cudaMalloc((void **) &gpu_cij_lib[i], 17770 * sizeof(float));
            cudaMemcpy(cij_lib[i], gpu_cij_lib[i], 17770 * sizeof(float),
                       cudaMemcpyHostToDevice);
        }
        cudaMalloc((void **) &gpu_data, sizeof(Data)));
        
        // Change into array.
        double *arr = new float[user_list_movies.size()];
        copy(user_list_movies.begin(), user_list_movies.end(), arr);
        
        float * gpu_user_list_movies;
        cudaMalloc((void **) &gpu_user_list_movies[i], user_list_movies.size() * sizeof(float));
        cudaMemcpy(arr, gpu_user_list_movies, user_list_movies.size() * sizeof(float),
                   cudaMemcpyHostToDevice);
    
        float *gpu_out_cij;
        
        cudaMalloc((void **) &gpu_out_cij, user_list_movies.size()/3 * sizeof(float));

        void cuda_get_cij_kernel(int user_id, int cur_movie_id, gpu_data, float * gpu_user_list_movies,
                                 float *gpu_out_cij, user_list_movies.size()/3, float **cij_lib, int alpha);
        
        cudaMemcpy(cij, gpu_out_cij, user_list_movies.size()/3 * sizeof(float),
                   cudaMemcpyDeviceToHost);
        //######################################################
        ////////////////////////////////////////////////////////
        sort(cij.begin(), cij.end());
        //######################################################
        float top = 0, bottom = 0;
        int loops = cij.size() > k ? k : cij.size();
        
        /////////////////////////////////////////////////////
        for (int i = 0; i < loops; i++)
        {
            //top += cij[cij.size() - 1 - i].second;
            //bottom += cij[cij.size() - 1 - i].first;
        }
        //######################################################
        float r_hat = bottom == 0? 3 : top/bottom;
        e_out += pow((val_ratings[i] - r_hat), 2);
    }
    e_out = pow(e_out / val_uid.size(), 0.5) ;
    cout << "error: "<<e_out<<endl;
    txtOut.close();
}

//This is where the execution of program begins
int main()
{
    Data data;
    
    //data.read_data("valid.txt", true);
    data.read_data("/Users/ziyanmo/outputs/small_train.txt", true);
    
    string ifile = "/Users/ziyanmo/outputs/small_probe.txt";
    string ofile = "/Users/ziyanmo/outputs/small_knn_probe.txt";
    
    
    float **cij_lib = new float*[17770];
    for (int i = 0; i < 17770; i++)
    {
        cij_lib[i] = new float[17770];
        for (int j = 0; j < 17770; j++)
        {
            cij_lib[i][j] = 0;
        }
    }
    int alpha = 300;
    int k = 30;
    auto start = high_resolution_clock::now();
    
    knn(&data, cij_lib, alpha, k, ifile, ofile, true);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop-start);
    cout << "duration: "<< duration.count() <<endl;
    for(int i = 0; i < 17770; ++i) {
        delete [] cij_lib[i];
    }
    delete [] cij_lib;
    
    return 0;
}

